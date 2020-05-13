from flask_material import Material
from flask import Flask, render_template, request, jsonify, redirect
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
#from flask_gtts import gtts
from gtts import gTTS
from vocabulary import Vocabulary
from googletrans import Translator
from datetime import datetime
import os
import torchvision.transforms as transforms
import io
import torch
import sys
import shutil
#import urllib


#print('beginning to download the file')
#url = 'https://dl.dropboxusercontent.com/s/8v30c2sbrdu7sbn/checkpoint_epoch21-step6471.pth.tar?dl=0'
#daFile=urllib.request.urlretrieve(url,'theModel2/dafile')
#file_name = 'theModel2/'
#g = urllib.request.urlopen(url)

#with open('theModel2/checkpoint_epoch21-step6471.pth.tar', 'b+w') as f:
#    f.write(g.read())



app = Flask(__name__)
Material(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///daDatabase.db'
db = SQLAlchemy(app)
upload_folder = '/static/uploadedImages'
app.config['upload_folder']=upload_folder
allowed_extensions = {'png','jpg','jpeg'}
vocab1 = Vocabulary(vocab_threshold=None,vocab_file='vocabulary/vocab.pkl', start_word='<start>',
end_word='<end>', unk_word='<unk>', vocab_from_file=True)


#asi funciona
#f = open('theModel/requirements.txt')
#print(f)

#print(os.getcwd())

from model2 import DecoderWithAttention, Encoder
encoder = Encoder()
decoder = DecoderWithAttention(attention_dim=512,embed_dim=512, decoder_dim=512, vocab_size=8855)
checkpoint = torch.load('theModel/checkpoint_epoch23-step12942.pth.tar', map_location=torch.device('cpu'))
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
encoder.eval()
decoder.eval()



@app.route('/aboutus', methods=['POST','GET'])
def getaboutus():
    if request.method == 'POST':
        pass
    else:
        texto = request.args.get('texto')
        return render_template('aboutus.html',tamanoTexto=texto)

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        pass
    else:
        texto = request.args.get('texto')
        if texto is None:
            return render_template('index.html', description='',daFileName='', tamanoTexto='noAgrandado')
        else:
            return render_template('index.html', description='',daFileName='', tamanoTexto=texto)
        


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route("/upload", methods=["GET","POST"])
def get_image():
    if request.method == 'POST':
        print('rquest value: ',request.form.get('elTexto'))
        print('request files: ',request.files)
        print('form: ',request.form)
        if 'elFile' not in request.files:
            print('no file uploaded')
            return redirect('/')
        file = request.files['elFile']


        if file.filename == '':
            print('No selected file')
            return redirect('/')
        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)
            print('the filename: ',filename)
            print(os.path.join(app.config['upload_folder'], filename))
            file.save(os.path.join('static','uploadedImages', filename))



            #pasar solo el filename a transform_image

            #img_bytes = file.read()
            #print('image bytes: ',str(img_bytes))
            daResult=predictCaption(filename)
            translator = Translator()
            translated = translator.translate(daResult, src='en', dest='es')

            #tts = gTTS(text=translated.text, lang='es')

            #print('the time now:',int(datetime.now().timestamp()))
            #nameOfAudioFile = str(int(datetime.now().timestamp()))+'.mp3'
            #tts.save('static/audio/'+nameOfAudioFile)

            
            
            return render_template('index.html', description='La foto ingresada contiene lo siguiente: '+translated.text,
            daFileName=filename, tamanoTexto=request.form.get('elTexto'))
            #return redirect('/')
        else:
            if not allowed_file(file.filename):
                return render_template('index.html', description='error: el archivo ingresado no es una imagen')

        #the image is saved in /static/uploadedImages


#necesito el vocabulario, neceisto el idx to word.



def predictCaption(filename):
    result=transform_image(filename)
    features = encoder(result)
    output = decoder.beamSearch(features)
    result2 = clean_sentence(output)
    print(result2)
    return result2

def clean_sentence(output):
    sentence = ''
    for i in output:
        word = vocab1.idx2word[i]
        if i==0:
            continue
        if i==1:
            break
        if i==18:
            sentence = sentence + word
        else:
            sentence = sentence + ' ' + word
    return sentence.strip()

from PIL import Image
def transform_image(filename):
    my_transforms = transforms.Compose([ 
    transforms.Resize(256),                          
    #transforms.RandomCrop(224), #se quita para hacer que la imagen siempre sea la misma.              
    transforms.RandomHorizontalFlip(),              
    transforms.ToTensor(),                           
    transforms.Normalize((0.485, 0.456, 0.406),      
                         (0.229, 0.224, 0.225))])

    image = Image.open(os.path.join('static/uploadedImages/',filename)).convert('RGB')
    return my_transforms(image).unsqueeze(0)

#funcion de prueba
@app.route('/predict', methods=['POST'])
def predict():
    #return 'this page will predict the image given'
    return jsonify({'imageId': '1', 'dateUploaded': '20/03/2020 17:14pm', 'prediction': 'a man wearing a suit'})


if __name__ == "__main__":
    app.run(debug=True)