import streamlit as st
import numpy as np
import tensorflow as tf
from lime import lime_text


st.title("Toxic Comment Classifier")

def load_model():
    return (tf.keras.models.load_model(filepath="toxicity_Embedding500000x32_Dropout_BiLSTM64_Dense128_Dropout_Dense256_Dense128_Dense6_betteremb.h5")
        , tf.keras.models.load_model(filepath='toxic_comment_vectorizer').layers[0])

model, vectorizer = load_model()

comment = st.text_input(label='give the input text here')

def toxic_level(val, minimum, p20, p40, p60, p80, maximum):
    if val >= minimum and val <p20:
        return 0
    elif val >=p20 and val < p40:
        return 1
    elif val >=p40 and val < p60:
        return 2
    elif val >=p60 and val < p80:
        return 3
    elif val >= p80 and val < maximum:
        return 4
    else:
        return -1

def lime_explainer(input_str):
    return model.predict(vectorizer(input_str))

text_explaination = None
toxic=0
severe_toxic=0
obscene=0
threat=0
insult=0
id_hate=0
explainer = lime_text.LimeTextExplainer()
if comment=="":
    st.write("Please write a comment")
else:
    pred = model.predict(np.expand_dims(vectorizer(comment), 0)).flatten()
    toxic, severe_toxic, obscene, threat, insult, id_hate = pred
    toxic = toxic_level(toxic,0, 0.2, 0.5, 0.7, 0.8, 1)
    severe_toxic = toxic_level(severe_toxic, 00, 0.2, 0.5, 0.7, 0.8, 1)
    obscene = toxic_level(obscene, 0, 0.2, 0.5, 0.7, 0.8, 1)
    threat = toxic_level(threat,  0, 0.2, 0.5, 0.7, 0.8, 1)
    insult = toxic_level(insult, 0, 0.2, 0.5, 0.7, 0.8, 1)
    id_hate = toxic_level(id_hate, 0, 0.2, 0.5, 0.7, 0.8, 1)
    st.write('Your comment is:')
    st.markdown(f"toxic: {toxic}/4 ")
    st.markdown(f"severely toxic: {severe_toxic}/4 ")
    st.markdown(f"obscene: {obscene}/4 ")
    st.markdown(f"threat: {threat}/4 ")
    st.markdown(f"insult: {insult}/4 ")
    st.markdown(f"identity hate{id_hate}/4 ")

if st.checkbox("Show why this is Toxic?"):
    show_toxic = st.empty()
    if toxic>1 or severe_toxic>0 or obscene>1 or threat>1 or insult>1 or id_hate>1:
        show_toxic.empty()
        waiting = st.info("Model loading...")
        toxic_explaination = explainer.explain_instance(comment, lime_explainer)
        waiting.empty()
        markdown_string = ""
        word_list = {k: v for k, v in toxic_explaination.as_list()}
        highest_toxic = min(word_list.values())
        least_toxic = max(word_list.values())
        for word in comment.split():
            # word = word.replace()
            if not word_list.get(word):
                markdown_string += f"{word} "
                continue
            toxic_red = (highest_toxic - word_list.get(word)) / (highest_toxic - least_toxic) * 255
            toxic_green = (word_list.get(word) - least_toxic) / (highest_toxic - least_toxic) * 255
            if toxic_red > 125:
                underline=True
            else:
                underline=False
            markdown_string += f"<span style='color:rgb({toxic_red},{toxic_green},0)'>{'<u>' if underline else ''}{word} {'</u>' if underline else ''}</span>"
        print(markdown_string)
        st.markdown(markdown_string, unsafe_allow_html=True)
        st.write("underlines words are the main cause for toxicity, red means toxic, green means nice, white means neutral.")
        st.write("A word being toxic only means that it is toxic in this context. Try a variation of the same sentence without the toxic words")
    else:
        show_toxic = st.info('This comment is not actually toxic :)')
