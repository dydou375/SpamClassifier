import streamlit as st
import regex as re
import pandas as pd
import joblib



def caracspec(text):
    caracspec = re.findall(r'[^a-zA-Z0-9\s!"(),-.:;?]', text)
    return len(caracspec)

def compter_chiffres(text):
    chiffres = re.findall(r'[0-9]', text)
    return len(chiffres)

def presence_redondance(text):
    redondance = re.findall(r'[^a-zA-Z0-9\s!"(),-.:;?]', text)
    return len(redondance)
    
# Fonction qui traite le dataframe en ajoutant les colonnes nécessaires
def ajouter_colonnes_message(df):
    # Ajout de la colonne longueur_message
    df['longueur_message'] = df['Contenu'].apply(len)
    # Ajout de la colonne caracspec_sms
    df['caracspec_sms'] = df['Contenu'].apply(caracspec)
    # Ajout de la colonne Tot_chiffres
    df['Tot_chiffres'] = df['Contenu'].apply(compter_chiffres)
    # Ajout de la colonne redondance_spam
    df['redondance_spam'] = df['Contenu'].apply(presence_redondance)

    return df[["longueur_message", "caracspec_sms", "Tot_chiffres", "redondance_spam"]]

def transformer_message_en_dataframe(message):
    # Créer un DataFrame avec le message
    df_message = pd.DataFrame({'Contenu': [message]})
    # Appliquer les transformations pour obtenir les colonnes nécessaires
    df_message = ajouter_colonnes_message(df_message)
    return df_message

#fonction qui permet de prédire si le message est spam ou ham
def prediction_spam(message_saisi):
    # Transformer le message en DataFrame
    df_message = transformer_message_en_dataframe(message_saisi)
    # Charger le modèle
    model = joblib.load('model_svc.joblib')
    resultat = model.predict(df_message)
    return resultat[0]


st.markdown("<h1 style='text-align: center;'>Spamlit</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Application de Détection de Spam par apprentissage automatique</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Spam, pas spam ? La question ne se pose plus ! Avec nous, la réponse est à portée de main !</h4>", unsafe_allow_html=True)

message = st.text_area("Saisissez un message :")
if message =="":
    st.write("Vous n'avez pas saisie de message")
else:
    prediction = prediction_spam(message)
    st.write(prediction)

## test pour savoir si le modèle fonctionne
spam = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
ham = "Hey, I'm just checking in to see if you've received my last message. How are you doing?"
print(prediction_spam(spam))
print(prediction_spam(ham))
