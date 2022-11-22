from classifier import image_data_pipeline,pred_label
from flask import Flask, render_template, request, Response, jsonify
import os
from flask_cors import CORS, cross_origin
from com_ineuron_utils.utils import decodeImage
import tensorflow as tf

app = Flask(__name__)




os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

CORS(app)




@app.route("/")
def home():
    # return "Landing Page"
    return render_template("index.html")


@app.route("/predict", methods=['POST', 'GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, 'file.jpg')
        test= image_data_pipeline('test',test_data=True)
        load_model = tf.keras.models.load_model("model")
        pred=load_model.predict(test)
        pred_l,percentage =pred_label(pred)
        result = pred_l+' : '+str(round(percentage,2))+'%'
        print(result)

    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"
    return jsonify(result)


# port = int(os.getenv("PORT"))
if __name__ == "__main__":
    port = 8000
    app.run(host='127.0.0.1', port=port)
