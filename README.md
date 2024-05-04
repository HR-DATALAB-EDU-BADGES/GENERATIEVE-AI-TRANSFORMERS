<img align="right" width="180" height="180" src="https://avatars.githubusercontent.com/u/115706761?s=400&u=7c6cae892816e172b0b7eef99f2d32adb948c6ad&v=4">

# GENERATIEVE AI: TRANSFORMER BASIS FUNCTIES

Een hands-on introductie van de meest gebruikte open-source Transformer-modellen en hun toepassingen in Natural Language Processing (NLP). 

Deze workshop is gebaseerd op de online beschikbare ["The Huggingface Course"](https://huggingface.co/learn/nlp-course/chapter1/).


## What kun je verwachten?

De cursus behandelt de basisfuncties van Transformer-modellen.
Je leert hoe je Transformer-modellen velig kunt gebruiken met de Hugging Face Transformers-bibliotheek.
Je oefent met verschillende NLP-taken om je vaardigheden te ontwikkelen.





|LEERDOEL|
| --------- |
| 1. [Coderen in Python met Anaconda + Jupyter notebooks](#l1) |
| 2. [NLP de Basis](#l2) |
| 3. [Tekstgegevens importeren en visualiseren](#l3)  | 
| 4. [Tekstgegevens voorbewerken en opschonen](#l4)  |
| 5. [Tekstgegevens analyseren en interpreteren](#l5)  |
| 6. [Aanpassen van vrije-tekst](#l6)   |

Na afloop van deze workshop ben je instaat om 
* met behulp van Python en Jupyter notebooks, tekstgegevens te importeren, te visualiseren en te voorbewerken. <br>
*  om met behulp van NLP technieken, tekstgegevens te analyseren en te interpreteren. <br>



********
# l1
## LEERDOEL [1]: Coderen in Python met Anaconda + Jupyter notebooks
******** 

<br>

|Materialen|
| --------- |
| [Jupyter-Notebook-for-Data-Science](https://github.com/HR-DATA-FABRIC/Jupyter-Notebook-for-Data-Science)| 

<br>

********
# l2
## NLP [2]: de bsis
******** 

## Natural Lanuage Processing [NLP] Defined

[Natural Language Processing (NLP)](https://www.ibm.com/cloud/learn/natural-language-processing) is a hybrid AI-discipline that is developed from [linguistics](https://en.wikipedia.org/wiki/Linguistics) and [computer science](https://en.wikipedia.org/wiki/Computer_science) to make human language intelligible to machines. 
The computers’ availability in the 1960s gave rise to NLP applications on computers known as [computational linguistics](https://en.wikipedia.org/wiki/Computational_linguistics). The structure of language is hierarchical comprising of seven levels each that constrain the use of computational linguistics. 

<br>


<div align="center">
    
    
level top-to-bottom  | Structure | refers to 
----- | ----------| --------------------------------------------------------
[1] | Phonology   | Elementary sounds
[2] | Morphology  | Elementary combinations of letters and sounds, called Morphemes
[3] | Lexical     | Individual words formed of Morphemes, called Lexemes
[4] | Syntax      | Combination of words, grammatical structure of a sentence
[5] | Semantics   | Rules used to convey meaning using the lower levels
[6] | Pragmatics  | Behavioral constraints on the use of a specific language
[7] | Discourse   | Multiple sentences together, rules about how they should relate to each other
    
</div>      

<br>

Syntactic ---[parsing](https://en.wikipedia.org/wiki/Parsing)--- and semantic ---[semiotics](https://en.wikipedia.org/wiki/Semiotics)--- analysis of text and speech to determine the meaning of a sentence. Syntax refers to the grammatical structure of a sentence, while semantics alludes to its intended meaning. By allowing computers to automatically analyze massive sets of data, NLP can find meaningful information in just milliseconds. 

 

<details>
 <summary><h3>NLP covers two application-domains NLU + NLG</h3></summary>

Natural Language Understanding [(NLU)](https://en.wikipedia.org/wiki/Natural-language_understanding): It is considered a "Hard AI-problem". The ambiguity and creativity of human language are just two of the characteristics that make NLP a demanding area to work in. The goal is to resolve ambiguities, obtain context and understand the meaning of what's being said. In particular, it tackles the complexities of language beyond the basic sentence structure. NLU is commonly used in [text mining](https://en.wikipedia.org/wiki/Text_mining) to understand consumer attitudes. In particular, sentiment analysis enables brands to monitor their customer feedback more closely, allowing them to cluster positive and negative social media comments and track net promoter scores. NLU can also establish a relevant [ontology](https://en.wikipedia.org/wiki/Ontology_(information_science)): a data structure which specifies the relationships between words and phrases. While humans naturally do this in [conversation](https://en.wikipedia.org/wiki/Discourse_analysis), the combination of these analyses is required for a machine to understand the intended meaning of different texts.
    
Natural Language Generation [(NLG)](https://en.wikipedia.org/wiki/Natural_language_generation): While NLU focuses on computers to comprehend human language, NLG enables computers to write. Initially, NLG systems used templates to generate text. Based on some data or query, an NLG system would fill in the blank, like a game of Mad Libs. But over time, natural language generation systems have evolved with the application of hidden Markov chains, recurrent neural networks, and transformers, enabling more dynamic text generation in real time. Given an internal representation, this involves selecting the right words, forming phrases and sentences. Sentences need to ordered so that information is conveyed correctly. It produces a human language text response based on some data input. This text can also be converted into a speech format through text-to-speech services.

</details>

NLU is about both analysis and synthesis ---understanding---.  Sentiment analysis and semantic search are examples of NLU. Captioning an image or video is mainly an NLG ---generating--- task since this type of input is not "textual". Text summarization and chatbot are applications that involve both [NLU + NLG](https://www.ibm.com/blogs/watson/2020/11/nlp-vs-nlu-vs-nlg-the-differences-between-three-natural-language-processing-concepts/). NLG also encompasses text summarization capabilities that generate summaries from input documents while maintaining the integrity of the information. 

***********

### Pre-Processing of free-texts & the NLP-data Pipeline

As mentioned earlier, NLP software typically analyzes text by breaking it up into
words (tokens) and sentences. Hence, any NLP pipeline has to start with a reliable
system to split the text into sentences (sentence segmentation) and further split a sentence
into words (word tokenization). On the surface, these seem like simple tasks,
and you may wonder why they need special treatment.

<img align="right" width="200" height="250" src="https://user-images.githubusercontent.com/684692/199322588-ba077b2e-8f09-4248-9259-0b61f77c28b1.png">


>NLP software typically works at the sentence level and
expects a separation of words at the minimum. So, we need some way to split a text
into words and sentences before proceeding further in a processing pipeline. Sometimes,
we need to remove special characters and digits, and sometimes, we don’t care
whether a word is in upper or lowercase and want everything in lowercase. Many
more decisions like this are made while processing text. Such decisions are addressed
during the pre-processing step of the NLP pipeline.

***********

### NLP OPEN-SOURCE Python Tools

To harnass NLP capabilities, there are high quality open-source NLP tools available allowing developers to discover valuable insights from unstructured texts.
That is, dealing with text analysis problems like classification, word ambiguity, sentiment analysis etc.

The here shown inventory is given on state-of-the-art  ---[Python programming language based](https://www.python.org/)--- open-source natural-language processing (NLP) tools & software. These are suites of libraries, frameworks, and applications for symbolic, statistical natural-language and speech processing.


Tool | NLP tasks | Distinctive features  | Neural networks | Best for | Not suitable for                          
--------|-----------|-----------------------|-----------------|----------|-----------------
NLTK    | Classification, tokenization, stemming. tagging. parsing. semantic reasoning | Over 50 corpora Package for chatbots Multilingual support| No | Training, Education, Research | Complex projects with large datasets      
Gensim | Text similarity. text summarization, SOTA topic modeling | Scalability and high performance Unsupervised training | No | Converting words and documents into vectors| Supervised text modeling Full NLP pipeline
SpaCy  | Tokenization, CNN tagging, parsing, named entity recognition. classification, sentiment analysis | 50+ languages (Dutch) available for tokenization Easy to learn and use | Yes |Teaching and research | Business production    
Textacy | Tokenization, Part-of-Speech Tagging, Dependency Parsing | High-performance SpaCy library | No | Access and extend spaCy’s core functionality | Beginners
Stanford CoreNLP Python Interface | Tokenization, multi- word-token expansion. lemmatization, POS tagging, dependency parsing | Different usage models Multilingual | Yes | Fully functional NLP systems, Co-reference resolution | Beginners                                 
Text Blob| POS tagging.noun phrase extraction sentiment analysis, classification, translation, spelling correction, etc. | Translation and spelling correction | No | NLP prototyping | Large scale productions § Altexsoft       
PyTorch-NLP | Word2Vector Encoding, Dataset Sampling | Neural Network pre-trained Embeddings | Yes | Rapid Prototyping, Research | Beginners
AllenNLP | high-level configuration language to implement many common approaches in NLP, such as transformer experiments, multi-task training, vision+language tasks, fairness, and interpretability | Solving natural language processing tasks in PyTorch |  Yes | Experimentation | Development has stopped
FlairNLP | Get insight from text extraction, word embedding, named entity recognition, parts of speech tagging, and text classification | Sense Disambiguation + Classification, Sentiment Analysis | No  | Supports Biomedical Datasets | Business production
Spark-NLP |  NLP-library for use with Apache Spark | Easy to scale by extending Apache Spark natively | Yes | Use of SOTA transformers such as BERT & ELMO at scale by extending Apache Spark natively | Beginners


***********


<br>


|Materialen|
| --------- |
| [ChatGPT uitgelegd](https://github.com/HR-ChatGPT/ChatGPT-UITGELEGD) |

<br>

********
# l3
## LEERDOEL [3]: Tekstgegevens kunnen importeren en visualiseren
******** 
