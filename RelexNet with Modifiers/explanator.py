import rdflib
import pandas as pd

from SPARQLWrapper import SPARQLWrapper
from regex import E


class Explanator():
    def __init__(self, graph_path, sentiwordnet_path=None):
        self.graph = rdflib.Graph()
        self.graph.parse(graph_path)
        for prefix, ns in self.graph.namespaces():
            if prefix == "xai-onto":
                self.onto_ns = rdflib.Namespace(ns)
            if prefix == "xai-data":
                self.data_ns = rdflib.Namespace(ns)
        self.construct_lexicon()
        self.construct_lexicon_score()
        self.construct_predicted_label()
        if sentiwordnet_path != None:
            self.init_sentiwordnet(sentiwordnet_path)

    def construct_lexicon(self):
        get_words_query = """
            SELECT DISTINCT ?lexicon 
            WHERE {
                ?wordid owl:hasValue ?lexicon .
                ?wordid a xai-onto:Word .
            }"""
        self.lexicons = [word[0] for word in self.graph.query(get_words_query)]
        self.lexicons.sort()

        for lexicon in self.lexicons:
            get_wordids_query = """
                SELECT ?wordid
                WHERE {
                    ?wordid owl:hasValue ?word .
                    FILTER (?word = \"""" + lexicon + """\"^^xsd:string).
                    ?wordid a xai-onto:Word .
                }
            """
            for wordid in self.graph.query(get_wordids_query):
                self.graph.add((
                    wordid[0],
                    rdflib.OWL.sameAs,
                    self.data_ns["Lexicon-{}".format(lexicon)]
                ))
                self.graph.add((
                    self.data_ns["Lexicon-{}".format(lexicon)],
                    rdflib.RDF.type,
                    self.onto_ns["Lexicon"]
                ))
                self.graph.add((
                    self.data_ns["Lexicon-{}".format(lexicon)],
                    rdflib.OWL.hasValue,
                    lexicon
                ))

    def get_lexicon_score(self, lexicon):
        query = """
            SELECT DISTINCT ?score ?class_id
            WHERE {
                ?evaluation xai-onto:output ?score.
                ?evaluation xai-onto:fromWord ?word.
                ?word owl:hasValue ?lexicon.
                ?evaluation xai-onto:ofLayer xai-data:layerL.
                ?evaluation xai-onto:classID ?class_id.
                FILTER (?lexicon = \"""" + lexicon + """\").
            }
            ORDER BY ASC(?class_id)
        """
        return self.graph.query(query)

    def construct_lexicon_score(self):
        for lexicon in self.lexicons:
            for score, class_id in self.get_lexicon_score(str(lexicon)):
                self.graph.add((
                    self.data_ns["Lexicon-{}-{}".format(lexicon, class_id)],
                    rdflib.RDF.type,
                    self.onto_ns["LexiconScore"]
                ))
                self.graph.add((
                    self.data_ns["Lexicon-{}-{}".format(lexicon, class_id)],
                    rdflib.OWL.hasValue,
                    score
                ))
                self.graph.add((
                    self.data_ns["Lexicon-{}-{}".format(lexicon, class_id)],
                    self.onto_ns["classID"],
                    class_id
                ))
                self.graph.add((
                    self.data_ns["Lexicon-{}-{}".format(lexicon, class_id)],
                    self.onto_ns["fromLexicon"],
                    self.data_ns["Lexicon-{}".format(lexicon)]
                ))

    def get_most_score_lexicons(self, sentence_id, class_id, threshold, limit):
        query = """
            SELECT ?lexicon_v ?lexiconScore_v
            WHERE {
                ?lexiconScore xai-onto:fromLexicon ?lexicon.
                ?lexiconScore owl:hasValue ?lexiconScore_v.
                ?lexicon owl:hasValue ?lexicon_v.
                ?lexiconScore xai-onto:classID ?class_id.
                ?word owl:sameAs ?lexicon.
                ?sentence xai-onto:hasWord ?word.
                ?sentence xai-onto:index ?sentence_id.
                FILTER (?class_id = """ + str(class_id) + """ && ?lexiconScore_v >= """ + str(threshold) + """ && ?sentence_id = """ + str(sentence_id) + """)
            }
            ORDER BY DESC(?lexiconScore_v)
            LIMIT """ + str(limit)
        lexicon_list=[]
        for value, score in self.graph.query(query):
            lexicon_list.append({"value": str(value), "score": float(score)})

        return lexicon_list

    def init_sentiwordnet(self, path):
        self.swn = pd.read_csv(path, sep="\t", header=25)
        self.swn.rename({"# POS": "POS"}, axis=1, inplace=True)
        self.swn["SynsetTerms"] = self.swn["SynsetTerms"].map(
            lambda s: list(map(lambda s: s.strip("#1234567890"), s.split())))
        self.swn["ID"] = self.swn["ID"].convert_dtypes()

    def get_sentiwordnet(self, lemma):
        terms = list(self.swn["SynsetTerms"])
        terms = [r for r in filter(lambda r: lemma in r[1], enumerate(terms))]
        return self.swn.iloc[[i for i, _ in terms]]

    def get_synsets(self, lemma):
        query = """
            PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
            SELECT DISTINCT ?synset
            FROM <http://wordnet-rdf.princeton.edu/data>
            WHERE {
                ?lemma ontolex:isLexicalizedSenseOf ?synset.
                FILTER CONTAINS (str(?lemma), \"/""" + lemma + """#\").
            }
        """
        sparql = SPARQLWrapper("http://rsmdb01.nci.org.au:8890/sparql")
        sparql.setQuery(query)
        sparql.setReturnFormat("json")
        result = sparql.queryAndConvert()
        result = list(
            map(lambda s: s["synset"]["value"], result["results"]["bindings"]))
        print(result)

    def get_hyponyms(self, synset):
        query = """
            PREFIX : <http://wordnet-rdf.princeton.edu/ontology#>
            SELECT ?h
            FROM <http://wordnet-rdf.princeton.edu/data>
            WHERE {
                ?s :hyponym ?h
                FILTER (?s = <""" + synset + """>)
            }
        """
        sparql = SPARQLWrapper("http://rsmdb01.nci.org.au:8890/sparql")
        sparql.setQuery(query)
        sparql.setReturnFormat("json")
        result = sparql.queryAndConvert()
        result = list(
            map(lambda s: s["h"]["value"], result["results"]["bindings"]))
        print(result)

    def get_datasets(self):
        query = """
            SELECT ?did ?dname
            WHERE {
                ?d xai-onto:index ?did.
                ?d xai-onto:name ?dname.
                ?d a xai-onto:Dataset
            }
        """
        dataset_list = []
        for did, dname in self.graph.query(query):
            dataset_list.append({"id": int(did), "name": str(dname)})
        return dataset_list

    def get_files(self, dataset_id):
        file_list = []
        query = """
            SELECT ?fid ?fn
            WHERE {
                ?d xai-onto:hasFile ?f.
                ?d xai-onto:index ?did.
                ?f xai-onto:index ?fid.
                ?f xai-onto:filePath ?fn.
                FILTER (?did = """ + str(dataset_id) + """).
            }
            ORDER BY ASC(?fid)
        """
        for fid, fn in self.graph.query(query):
            file_list.append({"id": int(fid), "path": str(fn)})
        return file_list

    def get_sentences(self, dataset_id, file_id):
        sentence_list = []
        query = """
            SELECT ?sid ?v ?label ?p_label
            WHERE {
                ?f xai-onto:hasSentence ?s.
                ?s xai-onto:index ?sid.
                ?s xai-onto:hasLabel ?label.
                ?s xai-onto:predictedLabel ?p_label.
                ?s owl:hasValue ?v.
                ?f xai-onto:index ?fid.
                ?d xai-onto:hasFile ?f.
                ?d xai-onto:index ?did.
                FILTER (?fid = """ + str(file_id) + """ && ?did = """ + str(dataset_id) + """).
            }
            ORDER BY ASC(?sid)
        """
        for sid, value, label, p_label in self.graph.query(query):
            sentence_list.append(
                {"id": int(sid), "value": str(value), "class_id": int(label), "predicted_class_id": int(p_label)})
        return sentence_list

    def get_words(self, dataset_id, file_id, sentence_id, context_window_size):
        word_list = []
        query = """
            SELECT ?wid ?v ?m
            WHERE {
                ?s xai-onto:hasWord ?w.
                ?w xai-onto:index ?wid.
                ?w owl:hasValue ?v.
                ?w xai-onto:isModifier ?m.
                ?s xai-onto:index ?sid.
                ?f xai-onto:hasSentence ?s.
                ?f xai-onto:index ?fid.
                ?d xai-onto:hasFile ?f.
                ?d xai-onto:index ?did.
                FILTER (?sid="""+str(sentence_id)+""" && ?fid="""+str(file_id)+""" && ?did="""+str(dataset_id)+""").
            }
            ORDER BY ASC(?wid)
        """
        for wid, v, m in self.graph.query(query):
            scores = []
            for s, cid in self.get_lexicon_score(v):
                scores.append(round(float(s), 2))
            word_list.append({
                "id": int(wid),
                "value": str(v),
                "isModifier": bool(m),
                "scores": scores
            })
        for i in range(len(word_list)):
            if word_list[i]["isModifier"]:
                j = 1
                while j <= context_window_size:
                    if i + j < len(word_list) and not word_list[i + j]["isModifier"]:
                        word_list[i + j]["isModifier"] = "Modified"
                    if i - j >= 0 and not word_list[i - j]["isModifier"]:
                        word_list[i - j]["isModifier"] = "Modified"
                    j += 1
        return word_list

    def get_word(self, dataset_id, file_id, sentence_id, word_id, context_window_size):
        words = self.get_words(dataset_id, file_id, sentence_id, context_window_size = context_window_size)
        for w in words:
            if w["id"] == int(word_id):
                return w

    def get_evaluations(self, dataset_id, file_id, sentence_id, word_id, layer):
        query = """
            SELECT DISTINCT ?class_id ?input ?output ?input_layer
            WHERE {
                ?e xai-onto:output ?output.
                ?e xai-onto:input ?input.
                ?e xai-onto:fromWord xai-data:"""+str(dataset_id)+"-"+str(file_id)+"-"+str(sentence_id)+"-"+str(word_id)+""".
                ?e xai-onto:ofInputLayer ?input_layer.
                ?e xai-onto:ofLayer xai-data:""" + layer + """.
                ?e xai-onto:classID ?class_id.
            }
            ORDER BY ASC(?class_id)
        """
        evals = []
        for cid, input, output, input_layer in self.graph.query(query):
            evals.append({"class_id": int(cid), "input": round(float(input), 2), "output": round(float(output), 2), "input_layer": str(input_layer)})
        return evals

    def construct_predicted_label(self):
        datasets = self.get_datasets()
        for dataset in datasets:
            dataset_id = dataset["id"]
            files = self.get_files(dataset_id)
            for file in files:
                file_id = file["id"]

                query = """
                    SELECT ?sid
                    WHERE {
                        ?f xai-onto:hasSentence ?s.
                        ?s xai-onto:index ?sid.
                        ?f xai-onto:index ?fid.
                        ?d xai-onto:hasFile ?f.
                        ?d xai-onto:index ?did.
                        FILTER (?fid = """+ str(file_id) +""" && ?did = """+ str(dataset_id) +""").
                    }
                """

                for sid in self.graph.query(query):
                    sentence_id = int(sid[0])
                    self.graph.add((
                        self.data_ns["{}-{}-{}".format(dataset_id,
                                                    file_id, sentence_id)],
                        self.onto_ns["predictedLabel"],
                        rdflib.Literal(self.get_predicted_label(
                            dataset_id, file_id, sentence_id), datatype=rdflib.XSD.int)
                    ))

    def get_predicted_label(self, dataset_id, file_id, sentence_id,):
        words = self.get_words(dataset_id, file_id, sentence_id, 0)
        max_word_id = max([word['id'] for word in words])
        evals = self.get_evaluations(dataset_id, file_id, sentence_id, max_word_id, "layerO")
        outputs = dict()
        for eval in evals:
            cid, _, output, _ = eval.values()
            outputs[cid] = output
        return max(outputs, key=outputs.get)

    def get_file_accuracy(self, dataset_id, file_id):
        sentences = self.get_sentences(dataset_id, file_id)
        err = 0
        for sentence in sentences:
            if sentence["class_id"] != sentence["predicted_class_id"]:
                err += 1
        return round(1 - err / len(sentences), 2)

    def get_model_config(self, model_id):
        query = """
            SELECT ?k ?v
            WHERE {
                ?m ?k ?v.
                ?m a xai-onto:Model.
                ?m xai-onto:index ?mid.
                FILTER (?mid = """+str(model_id)+""").
            }
        """
        conf = {}
        for key, val in self.graph.query(query):
            try:
                conf[str(key)] = int(val)
            except:
                pass
        return conf