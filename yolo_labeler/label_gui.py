import os
import sys
import shutil
from PySide6.QtWidgets import (QApplication, QMainWindow, QGraphicsView, 
                               QGraphicsScene, QGraphicsRectItem, QFileDialog, 
                               QVBoxLayout, QHBoxLayout, QPushButton, QWidget, 
                               QLabel, QInputDialog, QMessageBox)
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPixmap, QPen, QBrush, QColor, QAction

import numpy as np


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Speed Labeler - Team Edition")
        self.resize(1000, 700)
        
        self.team_id, ok = QInputDialog.getText(self, "Team Configuration", "Enter your Team Number:")
        if not ok or not self.team_id:
            self.team_id = "Unknown"


        self.image_list = []       # list of images in dir
        self.current_index = 0     
        self.current_image_path = None

        # setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # title
        self.info_label = QLabel("Open a Directory to Start")
        self.info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.info_label)

        self.view = QGraphicsView()
        self.scene = LabelingScene() 
        self.view.setScene(self.scene)
        layout.addWidget(self.view)

        # buttons
        button_layout = QHBoxLayout() # Horizontal layout for buttons
        
        self.btn_prev = QPushButton("<< Prev (A)")
        self.btn_prev.clicked.connect(self.prev_image)
        button_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("Next (D) >>")
        self.btn_next.clicked.connect(self.next_image)
        button_layout.addWidget(self.btn_next)
        
        layout.addLayout(button_layout)

        # more setup
        self.setup_menu()
        self.setup_shortcuts()

    def setup_shortcuts(self):
        # 'D'
        next_shortcut = QAction("Next", self)
        next_shortcut.setShortcut("D")
        next_shortcut.triggered.connect(self.next_image)
        self.addAction(next_shortcut)

        # 'A'
        prev_shortcut = QAction("Prev", self)
        prev_shortcut.setShortcut("A")
        prev_shortcut.triggered.connect(self.prev_image)
        self.addAction(prev_shortcut)

    def setup_menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        # open dir
        open_action = QAction("Open Directory", self)
        open_action.setShortcut("Ctrl+O") 
        open_action.triggered.connect(self.open_directory)
        file_menu.addAction(open_action)
        
        # save (manual)
        save_action = QAction("Save Label", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_current_labels)
        file_menu.addAction(save_action)
        
        # export
        export_action = QAction("Export for Submission", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_submission)
        file_menu.addAction(export_action)

    def open_directory(self):
        """
        Scans a folder for images and loads the first one.
        """
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        
        if folder_path:
            # check for proper format // NOTE: remove jpeg and bmp probably
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_list = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) 
                if f.lower().endswith(valid_extensions)
            ]
            
            self.image_list.sort()
            
            if not self.image_list:
                self.info_label.setText("No images found in that folder!")
                return
            
            # 3. Reset and Load First Image
            self.current_index = 0
            self.load_image_at_index(0)

    def load_image_at_index(self, index):
        """
        The Core Loader: clear scene, load pixels, update UI.
        """
        if 0 <= index < len(self.image_list):
            file_path = self.image_list[index]
            self.current_image_path = file_path
            
            # Reset Scene
            self.scene.clear()
            self.scene.stored_labels = [] 
            self.scene.current_rect_item = None
            
            # Load Pixels
            pixmap = QPixmap(file_path)
            self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())
            
            # Pass Data to Scene
            self.scene.image_width = pixmap.width()
            self.scene.image_height = pixmap.height()
            
            # --- THE FIX: Zoom out to fit the image in the window ---
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            # --------------------------------------------------------
            
            # Update UI Info
            self.info_label.setText(f"Image {index + 1} of {len(self.image_list)}: {os.path.basename(file_path)}")
            self.setWindowTitle(f"YOLO Labeler - {os.path.basename(file_path)}")

    def next_image(self):
        """
        Saves current work, then moves forward.
        """
        # 1. Auto-Save Logic
        self.save_current_labels()
        
        # 2. Move Index
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_image_at_index(self.current_index)
        else:
            self.info_label.setText("Finished! That was last the image. Export now with CMD + E.")

    def prev_image(self):
        """
        Saves current work (just in case), then moves backward.
        """
        self.save_current_labels()
        
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image_at_index(self.current_index)

    def save_current_labels(self):
        """
        Writes the .txt file for the CURRENT image.
        Can potentially label multiple objects
        """
        # Safety checks
        if not self.current_image_path: return
        labels = self.scene.stored_labels
        if not labels: return # Don't create empty files if no boxes drawn

        # Generate .txt path (image.jpg -> image.txt)
        base_name = os.path.splitext(self.current_image_path)[0]
        txt_path = base_name + ".txt"

        try:
            with open(txt_path, 'w') as f:
                for line in labels:
                    f.write(line + "\n")
            print(f"Auto-saved: {os.path.basename(txt_path)}")
        except Exception as e:
            print(f"Error saving: {e}")

    def export_submission(self):
        # CMD + E: saves renamed images and labels into user specified folder
        
        if not self.image_list:
            QMessageBox.warning(self, "Error", "No images loaded!")
            return

        # destination
        output_dir = QFileDialog.getExistingDirectory(self, "Select Folder to Save Submission (Downloads)")
        if not output_dir: return

        submission_folder = os.path.join(output_dir, f"Team{self.team_id}_Submission")
        os.makedirs(submission_folder, exist_ok=True)
        img_folder = os.path.join(submission_folder, "images")
        os.makedirs(img_folder, exist_ok=True)
        txt_folder = os.path.join(submission_folder, "labels")
        os.makedirs(txt_folder, exist_ok=True)

        success_count = 0
        
        # rename
        for index, src_img_path in enumerate(self.image_list):
            
            base_name = os.path.splitext(src_img_path)[0]
            src_txt_path = base_name + ".txt" # label file path
            
            # skip images without corresponding label
            if not os.path.exists(src_txt_path):
                print(f"Skipping index {index}: No label file found.")
                continue

            # ex: team5_image0.jpg, team5_image0.txt
            ext = os.path.splitext(src_img_path)[1] # .jpg or .png
            
            new_base_name = f"team{self.team_id}_image{index}"
            new_img_name = new_base_name + ext
            new_txt_name = new_base_name + ".txt"

            dst_img_path = os.path.join(img_folder, new_img_name)
            dst_txt_path = os.path.join(txt_folder, new_txt_name)

            # copy to new dir
            try:
                shutil.copy2(src_img_path, dst_img_path)
                shutil.copy2(src_txt_path, dst_txt_path)
                success_count += 1
            except Exception as e:
                print(f"Error exporting {new_base_name}: {e}")

        # SUCCESS
        QMessageBox.information(self, "Export Complete", 
                                f"Success! Exported {success_count} labeled pairs.\n\n"
                                f"Location: {submission_folder}\n"
                                f"Format: team{self.team_id}_imageN")




class LabelingScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_rect_item = None  
        self.start_point = None        # (x,y))
        self.image_width = 0        
        self.image_height = 0
        
        # Setup the "Pen" (The outline of your box)
        self.pen = QPen(Qt.red)
        self.pen.setWidth(2)
        
        # Setup the "Brush" (The fill of your box - Transparent)
        self.brush = QBrush(QColor(0, 0, 0, 0)) # r, g, b, alpha (0=transparent)
        
        # store labels
        self.stored_labels = []

    def mousePressEvent(self, event):
        """
        Triggered when you CLICK the mouse.
        GOAL: Start drawing a new box.
        """
        # 1. Get the click position relative to the SCENE (the image)
        pos = event.scenePos()
        
        self.start_point = pos
        
        self.current_rect_item = QGraphicsRectItem(QRectF(pos, pos))
        self.current_rect_item.setPen(self.pen)
        self.current_rect_item.setBrush(self.brush)
        
        self.addItem(self.current_rect_item)
        
        print(f"DEBUG: Mouse Clicked at {event.scenePos()}")
        super().mousePressEvent(event) # Keep this line to handle other events

    def mouseMoveEvent(self, event):
        """
        Triggered when you DRAG the mouse.
        GOAL: Update the size of the box as you move.
        """
        if self.current_rect_item:
            current_rect = QRectF(self.start_point, event.scenePos()).normalized()
            self.current_rect_item.setRect(current_rect)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Triggered when you LET GO of the mouse.
        GOAL: Finalize the box and calculate YOLO coordinates.
        """
        if self.current_rect_item:
            if self.image_width == 0:
                print("Error: No image loaded!")
                return
            
            rect = self.current_rect_item.rect()
            
            center_x = (rect.x() + rect.width() / 2) / self.image_width
            center_y = (rect.y() + rect.height() / 2) / self.image_height
            width = rect.width() / self.image_width
            height = rect.height() / self.image_height
    
            # store this in cooresponding label file for image
            label_str = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
            self.stored_labels.append(label_str)
            print(f"Captured: {label_str}")
            
            self.current_rect_item = None # Reset for next box
            
        super().mouseReleaseEvent(event)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())