class TextInputGetter:
    def __init__(self):
        self.texts = None

    def __getitem__(self, item):
        return self.texts[item]

    def __len__(self):
        return len(self.texts)


class ManualText(TextInputGetter):
    def __init__(self, texts: list[str]):
        super().__init__()
        self.texts = texts


class TextFileReader(TextInputGetter):
    def __init__(self, file_path: str):
        super().__init__()
        text_file = open(file_path, "r", encoding='utf8')
        self.texts = text_file.readlines()


class DocxFileReader(TextInputGetter):
    def __init__(self, file_path: str):
        import docx
        super().__init__()
        doc = docx.Document(file_path)
        self.texts = [para for para in doc.paragraphs]
