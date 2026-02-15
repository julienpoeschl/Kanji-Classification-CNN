import sys
import numpy as np
import torch
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL import Image
import json
import os


def main() -> None:
    # Load model and output map
    model_path = os.path.join("model", "checkpoints", "best_model.pt")
    map_path = os.path.join("dataset", "data", "processed_data", "processed_output_map.json")
    with open(map_path, "r", encoding="utf-8") as f:
        output_map = json.load(f)
    idx_to_class = {v: k for k, v in output_map.items()}

    from model.cnn_model import create_model
    num_classes = len(output_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=num_classes, device=device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class KanjiWindow(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Kanji Classification Screenshot")
            self.setGeometry(100, 100, 400, 500)
            layout = QVBoxLayout()

            self.img_label = QLabel("No image loaded")
            self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.img_label)

            btn_layout = QHBoxLayout()
            self.load_btn = QPushButton("Load Screenshot")
            self.load_btn.clicked.connect(self.load_image)
            btn_layout.addWidget(self.load_btn)
            layout.addLayout(btn_layout)

            self.result_label = QLabel("")
            self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.result_label)

            self.setLayout(layout)
            self.image = None

        def load_image(self):
            fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
            if fname:
                pixmap = QPixmap(fname)
                self.img_label.setPixmap(pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio))
                self.image = Image.open(fname).convert("L")
                self.run_inference()

        def preprocess(self, img):
            # Resize to 64x64, normalize to [0,1], add channel
            img = img.resize((64, 64))
            arr = np.array(img).astype(np.float32) / 255.0
            arr = arr[np.newaxis, :, :]  # (1, 64, 64)
            arr = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, 1, 64, 64)
            return arr

        def run_inference(self):
            if self.image is None:
                self.result_label.setText("No image loaded")
                return
            inp = self.preprocess(self.image)
            with torch.no_grad():
                logits = model(inp)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                top3 = np.argsort(probs)[-3:][::-1]
                results = [f"{idx_to_class[i]} ({probs[i]:.2%})" for i in top3]
                self.result_label.setText("Top 3 predictions:\n" + "\n".join(results))

    app = QApplication(sys.argv)
    window = KanjiWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()