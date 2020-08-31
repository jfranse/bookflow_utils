import mlflow

def set_note(note):
    mlflow.set_tag('mlflow.note.content', note)

def set_tags(tags):
    for key, value in tags.items():
            mlflow.set_tag(key, value)