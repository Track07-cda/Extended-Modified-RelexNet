from flask import Flask
from flask import render_template
from flask import request
import os
from explanator import Explanator

app = Flask(__name__)
sentiwordnet_path = "./data/SentiWordNet/SentiWordNet_3.0.0.txt"
explanator = None
config = None
try:
    explanator = Explanator("explaination_imdb.ttl", sentiwordnet_path)
    config = explanator.get_model_config(0)
except:
    pass


@app.route('/', methods=["POST", "GET"])
def index():
    global explanator
    error = None
    if request.method == "POST":
        if os.path.exists(request.form['DG_file_path']):
            explanator = Explanator(
                request.form['DG_file_path'], request.form['SWN_file_path'])
            config = explanator.get_model_config(0)
            print("DG loaded")
            return data()
        else:
            print("Invalid file path")
    return render_template("index.html")

@app.route('/data')
def data():
    datasets = explanator.get_datasets()
    return render_template("data.html", datasets = datasets)

@app.route('/data/<dataset_id>')
def dataset(dataset_id):
    files = explanator.get_files(dataset_id)
    return render_template("dataset.html", files=files, dataset_id=dataset_id)


@app.route('/data/<dataset_id>/<file_id>')
def file(dataset_id, file_id):
    sentences = explanator.get_sentences(dataset_id, file_id)
    accuracy = explanator.get_file_accuracy(dataset_id, file_id)

    class_num = config[str(explanator.onto_ns['classNum'])]
    count_p = [0 for _ in range(class_num)]
    count_gt = [0 for _ in range(class_num)]
    for sentence in sentences:
        count_p[sentence['predicted_class_id']] += 1
        count_gt[sentence['class_id']] += 1
    distribution = [{} for _ in range(class_num)]
    for class_id in range(class_num):
        distribution[class_id]['pridict'] = count_p[class_id] / sum(count_p) * 100
        distribution[class_id]['gt'] = count_gt[class_id] / sum(count_gt) * 100

    return render_template("file.html", sentences=sentences, dataset_id=dataset_id, file_id=file_id, accuracy = accuracy, class_num = class_num, distribution = distribution)


@app.route('/data/<dataset_id>/<file_id>/<sentence_id>')
def sentence(dataset_id, file_id, sentence_id):
    sentences = explanator.get_sentences(dataset_id, file_id)
    sentence = None
    for sent in sentences:
        if sent["id"] == int(sentence_id):
            sentence = sent
            break
    words = explanator.get_words(dataset_id, file_id, sentence_id, config[str(explanator.onto_ns["contextWindowSize"])])
    significant_lexicons = explanator.get_most_score_lexicons(sentence["id"], sentence["predicted_class_id"], 0.4, 10)
    print(words)
    print(significant_lexicons)
    print(sentence)
    return render_template("sentence.html", dataset_id=dataset_id, file_id=file_id, sentence=sentence, words=words, class_num = config[str(explanator.onto_ns["classNum"])], significant_lexicons = significant_lexicons)


@app.route('/data/<dataset_id>/<file_id>/<sentence_id>/<word_id>')
def word(dataset_id, file_id, sentence_id, word_id):
    word = explanator.get_word(dataset_id, file_id, sentence_id, word_id, config[str(explanator.onto_ns["contextWindowSize"])])
    layers = ["layerS", "layerB", "layerO"]
    evals = {}
    evaluation = dict()
    for layer in layers:
        evals[layer] = explanator.get_evaluations(dataset_id, file_id, sentence_id, word["id"], layer)
        evaluation[layer] = get_layer_output(evals[layer])

    evaluation["layerL"] = get_layer_input(evals["layerS"], "layerL")
    evaluation["layerM"] = get_layer_input(evals["layerS"], "layerM")
    evaluation["layerB_in"] = get_layer_input(evals["layerB"], "layerB")

    synsets = explanator.get_sentiwordnet(word["value"])
    return render_template("word.html", word=word, class_num=2, evaluation=evaluation, synsets = synsets)

@app.route('/data/<dataset_id>/<file_id>/<sentence_id>/<word_id>/<layer>/<class_id>')
def evaluation(dataset_id, file_id, sentence_id, word_id, layer, class_id):
    word = explanator.get_word(dataset_id, file_id, sentence_id, word_id, config[str(explanator.onto_ns["contextWindowSize"])])
    evaluation = explanator.get_evaluations(dataset_id, file_id, sentence_id, word_id, "layerS" if layer == "layerL" or layer == "layerM" else layer)
    class_id = int(class_id)
    i_layers = None
    output = None

    if layer == "layerL":
        i_layers = [{"name": "One-hot Encoder", "value": "OnehotEncoded "+word["value"]}]
        output = get_layer_input(evaluation, "layerL")[class_id]
        formula = "Linear layer"
    elif layer == "layerM":
        if word["isModifier"] == "Modified":
            i_layers = [{"name": "Modifier score from nearby modifier words", "value": ""}]
            output = get_layer_input(evaluation, "layerM")[class_id]
            formula = "Linear layer"
        else:
            i_layers = [{"name": "No Modifier", "value": 1}]
            output = 1
            formula = "Constant"
    elif layer == "layerS":
        i_layers = [{"name": "Layer L", "value": get_layer_input(evaluation, "layerL")[class_id]},
                    {"name": "Layer M", "value": get_layer_input(evaluation, "layerM")[class_id]}]
        output = get_layer_output(evaluation)[class_id]
        formula = "S = L * M"
    elif layer == "layerB":
        i_layers = [{"name": "Layer S", "value": get_layer_input(evaluation, "layerS")[class_id]},
                    {"name": "Layer B", "value": get_layer_input(evaluation, "layerB")[class_id]}]
        output = get_layer_output(evaluation)[class_id]
        formula = "B = B + S"
    elif layer == "layerO":
        i_layers = [{"name": "Layer B", "value": get_layer_input(evaluation, "layerB")[class_id]}]
        output = get_layer_output(evaluation)[class_id]
        formula = "O = softmax(B)"

    return render_template("evaluation.html", input_layers = i_layers, formula = formula, output = output, word = word, class_id = class_id)


def get_layer_output(evals):
    outputs = dict()
    for eval in evals:
        cid, _, output, _ = eval.values()
        outputs[cid] = output
    return [outputs[k] for k in sorted(outputs)]

def get_layer_input(evals, input_layer):
    inputs = dict()
    for eval in evals:
        cid, input_value, _, in_layer = eval.values()
        if in_layer[-6:] == input_layer:
            inputs[cid] = input_value
    return [inputs[k] for k in sorted(inputs)]