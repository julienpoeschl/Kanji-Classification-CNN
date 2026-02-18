from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL import Image
from controller import KanjiController, get_kanji_data

class KanjiWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kanji Classification Screenshot")
        self.setGeometry(100, 100, 400, 600)
        layout = QVBoxLayout()

        self.img_label = QLabel("No image loaded")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.img_label)

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Screenshot")
        self.load_btn.clicked.connect(self.load_image)
        btn_layout.addWidget(self.load_btn)
        self.paste_btn = QPushButton("Paste Screenshot")
        self.paste_btn.clicked.connect(self.paste_image)
        btn_layout.addWidget(self.paste_btn)

        layout.addLayout(btn_layout)

        self.pred_btns = []
        self.pred_btn_layout = QHBoxLayout()
        layout.addLayout(self.pred_btn_layout)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.setLayout(layout)
        self.image = None
        
        self.controller = KanjiController()
        self.kanji_data = get_kanji_data()
        self.idx_to_class = self.controller.get_idx_to_class()



    def paste_image(self):
        clipboard = QApplication.clipboard()

        if not clipboard:
            print("No clipboard found. Pasting image failed.")
            return
        
        mime = clipboard.mimeData()

        if not mime:
            print("No mime data found. Pasting image failed.")
            return

        if mime.hasImage():
            qimg = clipboard.image()
            pixmap = QPixmap.fromImage(qimg)
            self.img_label.setPixmap(pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio))
            # Convert QImage to PIL Image
            bits = qimg.bits()
            if not bits:
                print("No bits retreived from image. Pasting image failed.")
                return
            buffer = bits.asstring(qimg.width() * qimg.height() * qimg.depth() // 8)
            img = Image.frombytes("RGBA" if qimg.hasAlphaChannel() else "RGB", (qimg.width(), qimg.height()), buffer)
            if img.mode != "L":
                img = img.convert("L")
            self.image = img
            self.run_inference()
        else:
            self.info_label.setText("No image in clipboard.")


    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if fname:
            pixmap = QPixmap(fname)
            self.img_label.setPixmap(pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio))
            self.image = Image.open(fname).convert("L")
            self.run_inference()


    def run_inference(self):
        if self.image is None:
            self.info_label.setText("No image loaded")
            return
        top3, probs = self.controller.get_top_3_predictions(self.image)
        self.show_prediction_buttons(top3, probs)
        self.show_kanji_info(self.idx_to_class[top3[0]])

    def show_prediction_buttons(self, top3, probs):
        # Clear previous buttons
        for btn in self.pred_btns:
            btn.setParent(None)
        self.pred_btns = []
        for i in top3:
            kanji = self.idx_to_class[i]
            prob = probs[i]
            btn = QPushButton(f"{kanji} ({prob:.2%})")
            btn.clicked.connect(lambda _, k=kanji: self.show_kanji_info(k))
            self.pred_btn_layout.addWidget(btn)
            self.pred_btns.append(btn)

    def show_kanji_info(self, kanji):
        info = self.kanji_data.get(kanji)
        if not info:
            self.info_label.setText(f"No info found for {kanji}")
            return
        # Format info nicely
        text = f"<b>{kanji}</b><br>"
        for k, v in info.items():
            text += f"<b>{k}:</b> {v}<br>"
        self.info_label.setText(text)