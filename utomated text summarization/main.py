import tkinter as tk
from tkinter import scrolledtext
from tkinter import font
from PIL import Image, ImageTk
import numpy as np
import nltk
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')


def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


def extract_features_bert(sentences):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = model.encode(sentences)
    return embeddings


def cluster_sentences(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans


def summarize_with_bert(text, max_clusters=5):
    sentences = preprocess_text(text)
    if len(sentences) == 0:
        return "No sentences to summarize."

    embeddings = extract_features_bert(sentences)

    # Adjust the number of clusters to be at most the number of sentences
    n_clusters = min(max_clusters, len(sentences))

    kmeans = cluster_sentences(embeddings, n_clusters)

    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))

    summary_sentences = [sentences[int(i)] for i in sorted(avg)]
    summary = ' '.join(summary_sentences)
    return summary


def summarize_text():
    text = text_input.get("1.0", tk.END)
    summary = summarize_with_bert(text, max_clusters=5)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, summary)


def reset_input():
    text_input.delete("1.0", tk.END)
    output_box.delete("1.0", tk.END)


# Create the main window
root = tk.Tk()
root.title("Automated Text Summarization")

# Set window size
root.geometry("1080x1920")  # Set the width and height of the window

# Load background image
bg_image = Image.open("BG.png")
bg_image = bg_image.resize((1000, 800))  # Adjust the size as needed
background_image = ImageTk.PhotoImage(bg_image)

# Create a label to hold the background image
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create input box and label
text_input_label = tk.Label(root, text="Enter text to summarize:", height=2, width=20, fg="black", font=("Helvetica", 12, "bold"))
text_input_label.place(x=50, y=80)

text_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20)  # Increase width and height
text_input.place(x=50, y=120)

# Button to summarize text
summarize_button = tk.Button(root, text="Summarize Text", command=summarize_text,height=2, width=20,fg="black")
summarize_button.place(x=330, y=430)

# Create output box and label
output_box_label = tk.Label(root, text="Summary:",height=2, width=20,fg="black", font=("Helvetica", 12, "bold"))
output_box_label.place(x=50, y=520)

output_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20,fg="black")  # Increase width and height
output_box.place(x=50, y=560)

# Button to reset input
reset_button = tk.Button(root, text="Reset", command=reset_input,height=2, width=10,fg="black")
reset_button.place(x=900, y=760)


root.mainloop()
