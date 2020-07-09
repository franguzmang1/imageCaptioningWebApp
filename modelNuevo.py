# Basado en: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning (Recuperado en abril, 2020)
import torch
import torch.nn as nn
import torchvision.models as models
from math import log
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Encoder(nn.Module):
	"""
	Encoder for the model with attention.

	"""
	def __init__(self):
		super(Encoder, self).__init__()

		#como se hizo antes con arquitectura decoder encoder
		#resnet = models.resnet50(pretrained=True)

		#resnet152 = models.resnet152(pretrained=True)

		vgg16 = models.vgg16(pretrained=True)

		for param in vgg16.parameters():
			param.requires_grad_(False)


		#se quitan las 2 ultimas capas de resnet50
		modules2 = list(vgg16.children())[:-2][0][:-1]

		self.vgg16 = nn.Sequential(*modules2)

		self.entrenarvgg16()
		
		#self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size,encoded_image_size))


	def entrenarvgg16(self, entrenar=True):
		#todos las capas exceptuando la primera capa que es la que obtiene features muy sencillos como
		#lineas y bordes, sencillo y fundamental.
		for c in list(self.vgg16.children()):
			for p in c.parameters():
				p.requires_grad = entrenar



	def forward(self, images):
		"""
		Images are received in (batch_size,3,224,224)

		"""
		result = self.vgg16(images)#se pasa por resnet y se recibe 
		#(batch_size,2048,7,7)

		#esto antes se usaba, ahora no.
		#result = self.adaptive_pool(result)#de 7,7 ahora a 14,14. se repiten valores

		result = result.permute(0,2,3,1)#en vez de tener tensor en (batch_size,2048,14,14)
		#se pasa a tener shape (batch_size, 14, 14, 2048)

		#quitando esto a ver si no afecta
		#result = result.view(batch_size,-1,2048)

		#se devuelven los resultados del encoding en shape
		#(batch_size,196,2048)
		return result

	

class Attention(nn.Module):

	def __init__(self, encoder_dim, decoder_dim, attention_dim):
		""""
		encoder_dim: feature size of encoded images (512)
		decoder_dim: size of decoder's RNN (512)
		attention_dim: size of the attention network (512)
		"""
		super(Attention, self).__init__()
		self.encoder_att = nn.Linear(encoder_dim, attention_dim)
		#layer to transform decoder's output from 512 to 512
		self.decoder_att = nn.Linear(decoder_dim, attention_dim)
		#
		self.full_att = nn.Linear(attention_dim, 1)
		self.relu = nn.ReLU()
		self.softmax = nn.Softmax(dim=1)

		self.init_weights()


	def init_weights(self):
		torch.nn.init.xavier_uniform_(self.encoder_att.weight)
		torch.nn.init.xavier_uniform_(self.decoder_att.weight)
		torch.nn.init.xavier_uniform_(self.full_att.weight)
		self.encoder_att.bias.data.fill_(0)
		self.decoder_att.bias.data.fill_(0)
		self.full_att.bias.data.fill_(0)
		#encoder_out recibe los features de la foto
		#siempre son los mismos
		#decoder_hidden, hidden state del decoder.

		#23-01-19 07:31pm
		#aqui se tieenn que incluir calculos con los batch_sizes 
	def forward(self, encoder_out, decoder_hidden):
		"""
		encoder_out: fetures of images in shape: (batch_size,196,2048)
		decoder_hidden: hidden state of LSTMCell in shape: (batch_size,512)
		"""
		#se pasa de shape (batch_size,196,2048) a (batch_size,196,512)
		#los features se pasan de 2048 a 512 dimensiones
		att1 = self.encoder_att(encoder_out)

		att2 = self.decoder_att(decoder_hidden)

		#se suma lo que viene de encoder pasado por linear y en dimension 512
		#con lo que veine del decoder_hidden
		#a cada uno de los pixeles que tiene 512 dimensiones se le suma las 
		#512 dimensiones de lo que viene de hidden state de decoder
		#se obtiene shape (batch_size,196,512)

		#relu para aplicar sparcity
		#los valores que esten en negativo pasan a ser 0
		sum1 = self.relu(att1 + att2.unsqueeze(1))

		#se pasa de (batch_size, 196, 512) a (batch_size, 196)
		att = self.full_att(sum1).squeeze(2)#se tiene un score de importancia
		#de cada uno de los 196 pixeles

		#se pasa a softmax
		alpha = self.softmax(att)

		#se multiplica los features por alpha calculado (los pesos de los 196 pixeles)
		attention_weighted_encoding = (encoder_out*alpha.unsqueeze(2)).sum(dim=1)
		#shape resultante (batch_size,2048)


		return attention_weighted_encoding, alpha





class DecoderWithAttention(nn.Module):

	def __init__(self, attention_dim, embed_dim,decoder_dim, vocab_size, encoder_dim=512, dropout=0.5):
		"""
		:param attention_dim: size of attention network
		:param embed_dim: embedin size
		:param decoder_dim: size of decoder's RNN
		:param vocab_size: size of vocabulary
		:param encoder_dim: feature size of encoded images
		:param dropout: droput
		"""
		super(DecoderWithAttention, self).__init__()

		self.encoder_dim = encoder_dim
		self.attention_dim = attention_dim
		self.embed_dim = embed_dim
		self.decoder_dim = decoder_dim
		self.vocab_size = vocab_size
		self.dropout = dropout

		#se le pasa encoder_dim (2048), decoder_dim (512), attention_dim (512)
		self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

		#one vector for each word, the embedding dimension
		self.embedding = nn.Embedding(vocab_size, embed_dim)
		self.dropout = nn.Dropout(p=self.dropout)
		#embed_dim+encoder_dim porque recibe una palabras y la foto
		self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
		self.init_h = nn.Linear(encoder_dim,decoder_dim)#layer to find the initial hidden state of LSTMCell
		self.init_c = nn.Linear(encoder_dim,decoder_dim)#layer to find initial cell satte of LSTMCell
		self.f_beta = nn.Linear(decoder_dim,encoder_dim)#linear layer to create a sigmoid-activated gate
		self.sigmoid = nn.Sigmoid()
		self.fc = nn.Linear(decoder_dim, vocab_size)#linear layer to find scores over vocabulary
		self.init_weights()



	def init_weights(self):
		"""
		Inicializando con Xavier Initialization
		"""
		torch.nn.init.xavier_uniform_(self.embedding.weight)
		torch.nn.init.xavier_uniform_(self.init_h.weight)
		torch.nn.init.xavier_uniform_(self.init_c.weight)
		torch.nn.init.xavier_uniform_(self.f_beta.weight)
		torch.nn.init.xavier_uniform_(self.fc.weight)
		self.init_h.bias.data.fill_(0)
		self.init_c.bias.data.fill_(0)
		self.f_beta.bias.data.fill_(0)
		self.fc.bias.data.fill_(0)
		


	def init_hidden_state(self,encoder_out):
		"""
		Creates the initial hidden and cell states for the decoder's LSTM
		based on the encoded images.

		:param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
		:return: hidden state, cell state
		"""
		mean_encoder_out = encoder_out.mean(dim=1)
		h = self.init_h(mean_encoder_out)
		c = self.init_c(mean_encoder_out)
		return h,c


	def greedySearch(self, encoder_out):

		encoder_out = encoder_out.view(1,-1,512)
		predicted_sentence = []
		max_len = 20
		#se le pasa al embedding la palabra <start>
		inputs = self.embedding(torch.tensor(0).to(device))
		inputs = inputs.unsqueeze(0)
		h, c = self.init_hidden_state(encoder_out)

		for i in range(max_len):
			awe, _ = self.attention(encoder_out,h)
			gate = self.sigmoid(self.f_beta(h))
			awe = gate * awe

			h, c = self.decode_step(torch.cat([inputs, awe], dim=1), (h,c))

			preds = self.fc(h)
			preds = F.log_softmax(preds, dim=1)

			_, max_indice = torch.max(preds, dim=1)

			predicted_sentence.append(max_indice.cpu()[0].item())

			if max_indice == 1:
				break

			inputs = self.embedding(max_indice)

		return predicted_sentence


	def beamSearch(self, encoder_out):
		vocab_size=8855
		k=10#beam size
		encoder_out = encoder_out.view(1,-1,512)
		enc_image_size = int(math.sqrt(encoder_out.size(1)))
		encoder_dim = encoder_out.size(2)
		num_pixels = encoder_out.size(1)

		#expandir los features a 10 features iguales
		encoder_out = encoder_out.expand(k,num_pixels, encoder_dim)

		#vector donde guardar las palabras anteriores
		k_prev_words = torch.LongTensor([[0]]*k).to(device)

		#tensor para guardar las top k sequences
		seqs = k_prev_words

		#tensor para guardar los top k scores
		top_k_scores = torch.zeros(k,1).to(device)

		seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

		complete_seqs = list()
		complete_seqs_alpha = list()
		complete_seqs_scores = list()

		step = 1
		#devuelve 10 hidden states y 10 c de 512 cada uno
		h, c = self.init_hidden_state(encoder_out)

		while True:
			embeddings = self.embedding(k_prev_words).squeeze(1)#(s,embed_dim)
			awe, alpha = self.attention(encoder_out, h)#(s,encoder_dim), (s, num_pixels)

			#se hace reshape de los alphas, de (s,196) a (s, 14,14)
			alpha = alpha.view(-1, enc_image_size, enc_image_size)

			gate = self.sigmoid(self.f_beta(h)) #(s,encoder_dim)
			awe = gate * awe

			h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h,c))

			#se pasan lo que devuelve lstm por layer para que los ponga en dimension del vocabulary size
			scores = self.fc(h)
			#se pasa por softmax y por log
			scores = F.log_softmax(scores,dim=1)#(s, vocab_size)

			#se expanden los scores para que por cada uno de los 10 tops anteriores que hay
			#haya 8855 dimensiones, que representen la suma.luego se suma cada top score con uno de las palabras
			#posibles dadas en los scores.
			scores = top_k_scores.expand_as(scores) + scores

			if step == 1:#la primera vez solo se elige de la dimension 0. debido a que todas las dimensiones
			#representan a la misma palabra anterior. las 10 palabras anteriores son <start>
				top_k_scores, top_k_words = scores[0].topk(k,0,True,True)
			else:#se pone todas las 8855 predicciones de cada una de las 10 palabras en una dimension
			#y se busca las top 10
				top_k_scores, top_k_words = scores.view(-1).topk(k,0,True,True)

			#se divive porque si la palabra anterior era la 3era top entonces estara entre indices (8855*3)=26565
			#e indice (8855*4)=35420. esto indica indice con respecto a las top palabras guardadas
			prev_word_inds = top_k_words / vocab_size
			next_word_inds = top_k_words % vocab_size

			#add new words to sequences and alpha
			seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
			seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

			#cual sentence no esta completa
			incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if 
			next_word != 1]
			complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))


			if len(complete_inds) > 0:
				complete_seqs.extend(seqs[complete_inds].tolist())
				complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
				complete_seqs_scores.extend(top_k_scores[complete_inds])
			k -=len(complete_inds) #restarle al beam length el numero de secuencias completas


			if k ==0:#si numero de beam llego a 0
				break
			seqs = seqs[incomplete_inds]
			seqs_alpha = seqs_alpha[incomplete_inds]
			h = h[prev_word_inds[incomplete_inds]]
			c = c[prev_word_inds[incomplete_inds]]
			encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
			top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
			k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

			if step > 50:
				break
			step += 1

		i = complete_seqs_scores.index(max(complete_seqs_scores))
		seq = complete_seqs[i]
		alphas = complete_seqs_alpha[i]
		return seq


	def forward(self,encoder_out, encoded_captions):
		"""
		Forward propagation.

		:param encoder_out: encoded features. dimensions (batch_size,7,7,2048) 
		:param encoded_captions: encoded captions, a tensor of dimension (batch_size, number of words)
		(batch_size, max_caption_length)
		"""
		batch_size = encoder_out.size(0)
		encoder_dim = encoder_out.size(-1)




		#flatten the image
		encoder_out = encoder_out.view(batch_size, -1, encoder_dim)#shape will be (batch_size,49,2048)
		num_pixels = encoder_out.size(1)

		#se le quita la palabra <end> a los captions
		#encoded_captions0 = encoded_captions[:,:-1]

		#se pasa a tener las dimensiones de cada una de las palabras
		embeddings = self.embedding(encoded_captions[:,:-1])
		#(batch_size, caption_length, embeddingDimension)

		h, c = self.init_hidden_state(encoder_out)

		#create tensor to hold word prediction scores and alphas
		#predictions shape: (batch_szie, encodedcaption0size(1),vocabularysize)
		#numero de palabras sin <end>, nunca <end> va a ser un input.
		predictions = torch.zeros(batch_size, embeddings.size(1),self.vocab_size)
		predictions = predictions.to(device)
		#alphas shape: (batch_size, encoded_captions0.size(1), num_pixels)
		alphas = torch.zeros(batch_size, embeddings.size(1), num_pixels)
		alphas = alphas.to(device)
		#los alpha los saca attention da un score de importancia entre los 196
		#pixels

		for q in range(embeddings.size(1)):
			#se envian los 64 image features y 64 hidden states en todas las iteraciones
			attention_weighted_encoding, alpha = self.attention(encoder_out, h)

			#se pasa hidden states por layer que lo pasa a dimension 2048
			#se pasa los attention weights por un gate que tiene que ver con los 
			#hidden state de la lstm
			gate = self.sigmoid(self.f_beta(h))
			attention_weighted_encoding = gate * attention_weighted_encoding

			#se concatenan las palabras (512 dim) de los 64 batch size con los 2048 features
			#de los 64 bach size
			h, c = self.decode_step(torch.cat([embeddings[:,q,:], attention_weighted_encoding], dim=1), (h,c))
			#se aplica dropout para que no se dependa siempre de las mismas neuronas
			#se apagan algunas de las 512 dimensiones del hidden state, aplicando droput
			preds = self.fc(self.dropout(h))#se aplica tambien linear layer que pasa de 512 dimensiones 
			# a vocab size dimensiones
			predictions[:,q,:] = preds
			alphas[:,q,:] = alpha


		return predictions, alphas


#metodo para devolver oraciones limpias.
def clean_sentence(output):
    sentence = ''
    for i in output:
        word = data_loader.dataset.vocab.idx2word[i]
        if i == 0:
            continue
        if i == 1:
            break
        if i == 18:
            sentence = sentence + word
        else:
            sentence = sentence + ' ' + word
    return sentence.strip()


