import tkinter as tk
import tkinter.font as tf
from PIL import ImageTk, Image


class MyApp:
    def __init__(self, master, screnn_size):
        BUTTON_WIDTH = 10
        BUTTON_HEIGHT = 2
        WIDHT = int(screnn_size[0]/2)
        BUTTON_X = 3 * int(WIDHT / 2)
        BUTTON_Y = 800
        MARGIN = 15
        font = tf.Font(size=13, weight='bold')

        dobi = ImageTk.PhotoImage(Image.open('dobi.png'))
        captured_image_text = tk.Label(text='CAPTURE', font=font, bg='#7f8fa6')
        captured_image_text.place(x=MARGIN, y=MARGIN)
        captured_image = tk.Label()
        captured_image.image = dobi
        captured_image.configure(image=dobi)
        captured_image.place(x=MARGIN, y=MARGIN*3)

        result_image_text = tk.Label(text='RESULT', font=font, bg='#7f8fa6')
        result_image_text.place(x=WIDHT + MARGIN, y=MARGIN)
        result_image = tk.Label()
        result_image.image = dobi
        result_image.configure(image=dobi)
        result_image.place(x=WIDHT + MARGIN, y=MARGIN*3)

        # button
        capture_button = tk.Button(overrelief='solid', width=BUTTON_WIDTH, text='CAPTURE', bd=5, relief='raised', bg='#0097e6',
                                   fg='#dcdde1', height=BUTTON_HEIGHT, repeatdelay=1000, repeatinterval=100, compound=self.capture)
        capture_button.place(x=BUTTON_X, y=BUTTON_Y)
        run_button = tk.Button(overrelief='solid', width=BUTTON_WIDTH, text='RUN', bd=5, relief='raised', bg='#0097e6',
                               fg='#dcdde1', height=BUTTON_HEIGHT, repeatdelay=1000, repeatinterval=100, capture=self.run)
        run_button.place(x=BUTTON_X + 10*BUTTON_WIDTH + 5, y=BUTTON_Y)
        save_button = tk.Button(overrelief='solid', width=BUTTON_WIDTH, text='SAVE', bd=5, relief='raised', bg='#0097e6',
                                fg='#dcdde1', height=BUTTON_HEIGHT, repeatdelay=1000, repeatinterval=100, capture=self.save)
        save_button.place(x=BUTTON_X + 2*10*BUTTON_WIDTH + 5, y=BUTTON_Y)
        pass
        # self.F=tk.Frame(Parent)

        # self.F.pack()

       

        # self.button1=tk.Button(self.F)

        # self.button1["text"]="Hello, World!"

        # self.button1["background"]="green"

        # self.button1.pack()

def main():
    screnn_size = (1400, 850)
    screen_geometry = f'{screnn_size[0]}x{screnn_size[1]}+50+50'
    root = tk.Tk()
    root.title('BOOK_SCANNER')
    root.geometry(screen_geometry)
    root.iconbitmap(r'book.ico')
    root.configure(bg='#7f8fa6')
    root.resizable(False, False)
    myapp=MyApp(root, screnn_size)
    root.mainloop()



if __name__=="__main__":
    main()
