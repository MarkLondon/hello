import spacy

from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.training import train_data
iuiuiuuh
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    print('  ')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Load English language model
    nlp = spacy.blank('en')

    # Add NER pipeline component
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)

    # Add labels to NER model
    labels = ['PRODUCT_MODEL', 'BRAND', 'SERIES'] # Add any other relevant labels
    for label in labels:
        ner.add_label(label)

    # Train the NER model
    n_iter = 10
    optimizer = nlp.begin_training()
    for i in range(n_iter):
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            example = [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in zip(texts, annotations)]
            nlp.update(example, sgd=optimizer, losses=losses)
        print('Losses', losses)

    # Process the unseen data and extract the model names
    doc = nlp('Asus Vivobook Pro 15 OLED Creator Laptop | 15.6 Inch Full HD OLED Display | AMD Ryzen 5-5600H | 16GB RAM | 512GB SSD | AMD Radeon | Windows 11 | QWERTZ Keyboard | Quiet Blue')
    for ent in doc.ents:
        if ent.label_ == 'PRODUCT_MODEL':
            print(ent.text)
