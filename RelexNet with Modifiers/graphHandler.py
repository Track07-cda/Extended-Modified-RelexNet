from typing import Literal
import rdflib


class GraphHandler:
    def __init__(self, path, onto_prefix, data_prefix, model_config):
        self.graph = rdflib.Graph()
        self.graph.parse(path, format='ttl')
        for prefix, ns in self.graph.namespaces():
            if prefix == onto_prefix:
                self.onto_ns = rdflib.Namespace(ns)
            if prefix == data_prefix:
                self.data_ns = rdflib.Namespace(ns)
        self.eval_counter = {}
        self.model_counter = 0
        self.add_model_config(model_config)

    def add_model_config(self, config):
        self.graph.add((
            self.data_ns['model-{}'.format(self.model_counter)],
            rdflib.RDF.type,
            self.onto_ns['Model']
        ))
        self.graph.add((
            self.data_ns['model-{}'.format(self.model_counter)],
            self.onto_ns['index'],
            rdflib.Literal(self.model_counter, datatype=rdflib.XSD.int)
        ))
        self.graph.add((
            self.data_ns['model-{}'.format(self.model_counter)],
            self.onto_ns['implements'],
            self.data_ns['relexnet']
        ))
        for cfg in config.keys():
            self.graph.add((
                self.data_ns['model-{}'.format(self.model_counter)],
                self.onto_ns[cfg],
                rdflib.Literal(config[cfg], datatype=rdflib.XSD.int)
            ))
        self.model_counter += 1

    def get_eval_URIRef(self, dataset_id, file_id, sentence_id, word_id, eval_id):
        return self.data_ns["{}-{}-{}-{}-{}".format(dataset_id, file_id, sentence_id, word_id, eval_id)]

    def get_word_URIRef(self, dataset_id, file_id, sentence_id, word_id):
        return self.data_ns["{}-{}-{}-{}".format(dataset_id, file_id, sentence_id, word_id)]

    def get_sentence_URIRef(self, dataset_id, file_id, sentence_id):
        return self.data_ns["{}-{}-{}".format(dataset_id, file_id, sentence_id)]

    def get_file_URIRef(self, dataset_id, file_id):
        return self.data_ns["{}-{}".format(dataset_id, file_id)]

    def get_dataset_URIRef(self, dataset_id):
        return self.data_ns[str(dataset_id)]

    def add_dataset(self, dataset_id, name):
        self.eval_counter[dataset_id] = {}
        self.graph.add((
            self.get_dataset_URIRef(dataset_id),
            rdflib.RDF.type,
            self.onto_ns['Dataset']
        ))
        self.graph.add((
            self.get_dataset_URIRef(dataset_id),
            self.onto_ns['index'],
            rdflib.Literal(dataset_id, datatype=rdflib.XSD.int)
        ))
        self.graph.add((
            self.get_dataset_URIRef(dataset_id),
            self.onto_ns['name'],
            rdflib.Literal(name, datatype=rdflib.XSD.string)
        ))

    def add_file(self, dataset_id, file_id, file_name):
        if self.get_dataset_URIRef(dataset_id) in self.graph.subjects():
            self.eval_counter[dataset_id][file_id] = {}
            self.graph.add((
                self.get_file_URIRef(dataset_id, file_id),
                rdflib.RDF.type,
                self.onto_ns['File']
            ))
            self.graph.add((
                self.get_dataset_URIRef(dataset_id),
                self.onto_ns['hasFile'],
                self.get_file_URIRef(dataset_id, file_id),
            ))
            self.graph.add((
                self.get_file_URIRef(dataset_id, file_id),
                self.onto_ns['index'],
                rdflib.Literal(file_id, datatype=rdflib.XSD.int)
            ))
            self.graph.add((
                self.get_file_URIRef(dataset_id, file_id),
                self.onto_ns['filePath'],
                rdflib.Literal(file_name, datatype=rdflib.XSD.string)
            ))
            if file_id > 0:
                self.graph.add((
                    self.get_file_URIRef(dataset_id, file_id),
                    self.onto_ns['hasPrevious'],
                    self.get_file_URIRef(dataset_id, file_id - 1)
                ))
                self.graph.add((
                    self.get_file_URIRef(dataset_id, file_id - 1),
                    self.onto_ns['hasNext'],
                    self.get_file_URIRef(dataset_id, file_id)
                ))

    def add_sentence(self, dataset_id, file_id, sentence_id, text, label=None):
        if self.get_file_URIRef(dataset_id, file_id) in self.graph.subjects():
            self.eval_counter[dataset_id][file_id][sentence_id] = {}
            self.graph.add((
                self.get_sentence_URIRef(dataset_id, file_id, sentence_id),
                rdflib.RDF.type,
                self.onto_ns['Sentence']
            ))
            self.graph.add((
                self.get_file_URIRef(dataset_id, file_id),
                self.onto_ns['hasSentence'],
                self.get_sentence_URIRef(dataset_id, file_id, sentence_id)
            ))
            self.graph.add((
                self.get_sentence_URIRef(dataset_id, file_id, sentence_id),
                self.onto_ns['index'],
                rdflib.Literal(sentence_id, datatype=rdflib.XSD.int)
            ))
            self.graph.add((
                self.get_sentence_URIRef(dataset_id, file_id, sentence_id),
                rdflib.OWL.hasValue,
                rdflib.Literal(text, datatype=rdflib.XSD.string)
            ))
            if sentence_id > 0:
                self.graph.add((
                    self.get_sentence_URIRef(dataset_id, file_id, sentence_id),
                    self.onto_ns['hasPrevious'],
                    self.get_sentence_URIRef(
                        dataset_id, file_id, sentence_id - 1)
                ))
                self.graph.add((
                    self.get_sentence_URIRef(
                        dataset_id, file_id, sentence_id - 1),
                    self.onto_ns['hasNext'],
                    self.get_sentence_URIRef(dataset_id, file_id, sentence_id)
                ))
            if label != None:
                self.graph.add((
                    self.get_sentence_URIRef(dataset_id, file_id, sentence_id),
                    self.onto_ns['hasLabel'],
                    rdflib.Literal(label, datatype=rdflib.XSD.int)
                ))

    def add_word(self, dataset_id, file_id, sentence_id, word_id, value, is_modifier):
        if self.get_sentence_URIRef(dataset_id, file_id, sentence_id) in self.graph.subjects():
            self.eval_counter[dataset_id][file_id][sentence_id][word_id] = 0
            self.graph.add((
                self.get_word_URIRef(dataset_id, file_id,
                                     sentence_id, word_id),
                rdflib.RDF.type,
                self.onto_ns['Word']
            ))
            self.graph.add((
                self.get_word_URIRef(dataset_id, file_id,
                                     sentence_id, word_id),
                self.onto_ns['index'],
                rdflib.Literal(word_id, datatype=rdflib.XSD.int)
            ))
            self.graph.add((
                self.get_word_URIRef(dataset_id, file_id,
                                     sentence_id, word_id),
                rdflib.OWL.hasValue,
                rdflib.Literal(value, datatype=rdflib.XSD.string)
            ))
            self.graph.add((
                self.get_sentence_URIRef(dataset_id, file_id, sentence_id),
                self.onto_ns['hasWord'],
                self.get_word_URIRef(dataset_id, file_id, sentence_id, word_id)
            ))
            self.graph.add((
                self.get_word_URIRef(dataset_id, file_id,
                                     sentence_id, word_id),
                self.onto_ns['isModifier'],
                rdflib.Literal(is_modifier, datatype=rdflib.XSD.boolean)
            ))
            if word_id > 0:
                self.graph.add((
                    self.get_word_URIRef(
                        dataset_id, file_id, sentence_id, word_id),
                    self.onto_ns['hasPrevious'],
                    self.get_word_URIRef(
                        dataset_id, file_id, sentence_id, word_id - 1)
                ))
                self.graph.add((
                    self.get_word_URIRef(
                        dataset_id, file_id, sentence_id, word_id - 1),
                    self.onto_ns['hasNext'],
                    self.get_word_URIRef(
                        dataset_id, file_id, sentence_id, word_id)
                ))

    def add_evaluation(self, dataset_id, file_id, sentence_id, word_id, class_id, layer, input, output, input_layer):
        if self.get_word_URIRef(dataset_id, file_id, sentence_id, word_id) in self.graph.subjects():
            eval_id = self.eval_counter[int(dataset_id)][int(
                file_id)][int(sentence_id)][int(word_id)]
            self.eval_counter[int(dataset_id)][int(
                file_id)][int(sentence_id)][int(word_id)] += 1
            self.graph.add((
                self.get_eval_URIRef(dataset_id, file_id,
                                     sentence_id, word_id, eval_id),
                rdflib.RDF.type,
                self.onto_ns['Evaluation']
            ))
            self.graph.add((
                self.get_eval_URIRef(dataset_id, file_id,
                                     sentence_id, word_id, eval_id),
                self.onto_ns['index'],
                rdflib.Literal(eval_id, datatype=rdflib.XSD.int)
            ))
            self.graph.add((
                self.get_eval_URIRef(dataset_id, file_id,
                                     sentence_id, word_id, eval_id),
                self.onto_ns['ofLayer'],
                self.data_ns[layer]
            ))
            self.graph.add((
                self.get_eval_URIRef(dataset_id, file_id,
                                     sentence_id, word_id, eval_id),
                self.onto_ns['fromWord'],
                self.get_word_URIRef(dataset_id, file_id, sentence_id, word_id)
            ))
            self.graph.add((
                self.get_eval_URIRef(dataset_id, file_id,
                                     sentence_id, word_id, eval_id),
                self.onto_ns['classID'],
                rdflib.Literal(class_id, datatype=rdflib.XSD.int)
            ))
            if input != None:
                self.graph.add((
                    self.get_eval_URIRef(
                        dataset_id, file_id, sentence_id, word_id, eval_id),
                    self.onto_ns['input'],
                    rdflib.Literal(input, datatype=rdflib.XSD.float)
                ))
            self.graph.add((
                self.get_eval_URIRef(dataset_id, file_id,
                                     sentence_id, word_id, eval_id),
                self.onto_ns['output'],
                rdflib.Literal(output, datatype=rdflib.XSD.float)
            ))
            if input_layer != None:
                self.graph.add((
                    self.get_eval_URIRef(
                        dataset_id, file_id, sentence_id, word_id, eval_id),
                    self.onto_ns['ofInputLayer'],
                    self.data_ns[input_layer]
                ))
