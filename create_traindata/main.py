import cv2
import hydra
from omegaconf import DictConfig
import os
import time
from datetime import datetime
from pathlib import Path

class DataCollector:
    def __init__(self, target_name, anomaly_class, crop_size=None, camera_id=0):
        self.root_dir = Path(target_name)
        self.anomaly_class = anomaly_class
        self.camera_id = camera_id
        self.crop_size = crop_size # [width, height] or None
        
        # ディレクトリ設定
        self.dirs = {
            '1': self.root_dir / 'train' / 'good',
            '2': self.root_dir / 'test' / 'good',
            '3': self.root_dir / 'test' / self.anomaly_class
        }
        
        # ディレクトリ作成
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
            
        # 状態管理
        self.current_mode = '1'  # 1: train/good, 2: test/good, 3: test/anomaly
        self.last_save_time = 0.0
        self.roi_start = None
        self.roi_end = None
        self.drawing = False
        self.roi_selected = False
        self.save_count = 0
        
        # マウスイベント用
        self.ix, self.iy = -1, -1
        
    def mouse_callback(self, event, x, y, flags, param):
        if self.crop_size:
            return # 中央固定なのでマウス操作は無効

        # --- 自由切り抜きモード ---
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.roi_start = (x, y)
            self.roi_end = (x, y)
            self.roi_selected = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.roi_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.roi_end = (x, y)
            # 始点と終点を正規化
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            self.roi_start = (min(x1, x2), min(y1, y2))
            self.roi_end = (max(x1, x2), max(y1, y2))
            
            if (self.roi_end[0] - self.roi_start[0] > 10) and (self.roi_end[1] - self.roi_start[1] > 10):
                self.roi_selected = True
            else:
                self.roi_selected = False

    def get_save_path(self):
        return self.dirs[self.current_mode]

    def save_image(self, frame):
        if not self.roi_selected:
            return False
            
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        
        # 画面外にはみ出している場合の処理（クリッピング）
        h, w = frame.shape[:2]
        
        cx1 = max(0, min(x1, w))
        cy1 = max(0, min(y1, h))
        cx2 = max(0, min(x2, w))
        cy2 = max(0, min(y2, h))
        
        if cx2 <= cx1 or cy2 <= cy1:
            return False
            
        crop_img = frame[cy1:cy2, cx1:cx2]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.png"
        save_path = self.get_save_path() / filename
        
        try:
            cv2.imwrite(str(save_path), crop_img)
            print(f"Saved: {save_path}")
            self.save_count += 1
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        window_name = "Data Collector"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("=== Controls ===")
        if self.crop_size:
            print(f"Fixed Center Crop Mode: {self.crop_size}")
            print("ROI is fixed at the center of the screen.")
        else:
            print("Mouse Drag: Select ROI (Red Box)")
            
        print("s: Shot (Save 1 image)")
        print("c: Hold to Burst Shot (1 sec interval)")
        print("1: Switch to Train/Good")
        print("2: Switch to Test/Good")
        print("3: Switch to Test/Anomaly")
        print("q: Quit")
        print("================")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            display_frame = frame.copy()
            
            # 固定サイズモードの場合、中央にROIを設定
            if self.crop_size:
                h, w = frame.shape[:2]
                cw, ch = self.crop_size
                cx, cy = w // 2, h // 2
                x1 = cx - cw // 2
                y1 = cy - ch // 2
                x2 = x1 + cw
                y2 = y1 + ch
                
                self.roi_start = (int(x1), int(y1))
                self.roi_end = (int(x2), int(y2))
                self.roi_selected = True

            # ROI描画
            if self.roi_selected and self.roi_start and self.roi_end:
                pt1 = self.roi_start
                pt2 = self.roi_end
                cv2.rectangle(display_frame, pt1, pt2, (0, 0, 255), 2)

            # ステータス表示
            mode_name = {
                '1': 'Train/Good',
                '2': 'Test/Good',
                '3': f'Test/{self.anomaly_class}'
            }[self.current_mode]
            
            status_color = (0, 255, 0) if self.current_mode in ['1', '2'] else (0, 0, 255)
            
            cv2.putText(display_frame, f"Mode: {mode_name} (Key 1-3)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # 連続撮影ステータス（押されている間だけ表示する術がないので説明を表示）
            cv2.putText(display_frame, "Hold 'c' for Burst (1s)", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(display_frame, f"Saved: {self.save_count}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if not self.roi_selected:
                 cv2.putText(display_frame, "Please Select ROI", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.roi_selected:
                    self.save_image(frame)
                else:
                    print("Please select ROI first.")
            elif key == ord('c'):
                # 押している間、1秒間隔で保存
                if self.roi_selected:
                    current_time = time.time()
                    if current_time - self.last_save_time >= 1.0:
                        if self.save_image(frame):
                            self.last_save_time = current_time
            elif key in [ord('1'), ord('2'), ord('3')]:
                self.current_mode = chr(key)
                print(f"Switched storage to: {self.dirs[self.current_mode]}")

        cap.release()
        cv2.destroyAllWindows()

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    # crop_sizeの取得（設定がない場合はNone）
    crop_size = None
    if "crop_size" in cfg and cfg.crop_size:
        crop_size = tuple(cfg.crop_size) # ListConfig -> tuple
        
    collector = DataCollector(cfg.target, cfg.anomaly, crop_size, cfg.camera)
    collector.run()

if __name__ == "__main__":
    main()
