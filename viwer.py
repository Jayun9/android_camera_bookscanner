import tkinter as tk
import tkinter.font as tf
from tkinter.simpledialog import askstring
from PIL import ImageTk, Image
import threading
from imgprocess import process
import matplotlib.pyplot as plt



class MyApp:
    def __init__(self, master, screnn_size):
        self.save_folder_name = None
        url = askstring("IP", "What is your Ip adress?")
        full_url = f'https://{url}/video'
        BUTTON_WIDTH = 10
        BUTTON_HEIGHT = 2
        WIDHT = int(screnn_size[0]/2)
        BUTTON_X = 3 * int(WIDHT / 2)
        BUTTON_Y = 800
        MARGIN = 15
        self.pro = process(full_url)
        font = tf.Font(size=13, weight='bold')

        dobi = ImageTk.PhotoImage(Image.open('dobi.png'))
        captured_image_text = tk.Label(text='CAPTURE', font=font, bg='#7f8fa6')
        captured_image_text.place(x=MARGIN, y=MARGIN)
        self.captured_image = tk.Label()
        self.captured_image.image = dobi
        self.captured_image.configure(image=dobi)
        self.captured_image.place(x=MARGIN, y=MARGIN*3)

        result_image_text = tk.Label(text='RESULT', font=font, bg='#7f8fa6')
        result_image_text.place(x=WIDHT + MARGIN, y=MARGIN)
        self.result_image = tk.Label()
        self.result_image.image = dobi
        self.result_image.configure(image=dobi)
        self.result_image.place(x=WIDHT + MARGIN, y=MARGIN*3)

        # button
        capture_button = tk.Button(master, overrelief='solid', width=BUTTON_WIDTH, text='CAPTURE', bd=5, relief='raised', bg='#0097e6',
                                   fg='#dcdde1', height=BUTTON_HEIGHT, repeatdelay=1000, repeatinterval=100, command=self.capture)
        capture_button.place(x=BUTTON_X, y=BUTTON_Y)
        run_button = tk.Button(master, overrelief='solid', width=BUTTON_WIDTH, text='RUN', bd=5, relief='raised', bg='#0097e6',
                               fg='#dcdde1', height=BUTTON_HEIGHT, repeatdelay=1000, repeatinterval=100, command=self.run)
        run_button.place(x=BUTTON_X + 10*BUTTON_WIDTH + 5, y=BUTTON_Y)
        save_button = tk.Button(master, overrelief='solid', width=BUTTON_WIDTH, text='SAVE', bd=5, relief='raised', bg='#0097e6',
                                fg='#dcdde1', height=BUTTON_HEIGHT, repeatdelay=1000, repeatinterval=100, command=self.save)
        save_button.place(x=BUTTON_X + 2*10*BUTTON_WIDTH + 5, y=BUTTON_Y)
        self.t1 = threading.Thread(target=self.pro.stream)
        self.t1.start()

    def __del__(self):
        self.pro.stream_stop=True 

    def upload_image_to_tkinter(self, label, img, *place):
        axis = place
        label.image= img
        label.configure(image=img)
        if axis != ():
            label.place(x=axis[0], y=axis[1])

    def capture(self):
        capture_array = self.pro.capture()
        capture_image = ImageTk.PhotoImage(image=Image.fromarray(capture_array))
        self.upload_image_to_tkinter(self.captured_image, capture_image)

    def run(self):
        result_img = self.pro.run()
        self.upload_image_to_tkinter(self.result_image, result_img)
        
    def save(self):
        if self.save_folder_name is None:
            self.save_folder_name = askstring("SAVE", "Enter the name of the folder to be saved")
        self.pro.save(self.save_folder_name)

def main():
    # 192.168.42.129:8080
    screnn_size = (1400, 900)
    screen_geometry = f'{screnn_size[0]}x{screnn_size[1]}+50+50'
    root = tk.Tk()
    root.title('BOOK_SCANNER')
    root.geometry(screen_geometry)
    root.iconbitmap(r'book.ico')
    root.configure(bg='#7f8fa6')
    root.resizable(False, False)
    MyApp(root, screnn_size)
    root.mainloop()


if __name__=="__main__":
    main()
