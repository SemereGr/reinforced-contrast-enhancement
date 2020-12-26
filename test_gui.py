"""Designing GUI with tkinter"""
import tkinter as tk
from PIL import ImageTk, Image

class GUI(tk.Tk, object):
	"""Models gui environment"""
	def __init__(self):
		super().__init__()
		self.option_add('*Font', 'Times')
		self.title('Reinforcement Learning Project')
		self.rate: int = None
		self.flag: bool = False
		self.notify = tk.StringVar(value='Deteriorated Image')
		self.build_gui()

	def build_gui(self):
		"""builds the GUI"""
		#create window object

		#declare all window objects here
		self.space = tk.Label(self, text = '    ')
		self.space.grid(row=0, column=12, rowspan=1)

		self.text1 = tk.Label(self, textvariable = self.notify, borderwidth=2,relief="groove")
		self.text1.grid(row=1, column=6)

		self.text2 = tk.Label(self, text = 'Modified Image', borderwidth=2, relief="groove")
		self.text2.grid(row=1, column=20, columnspan=6)

		self.space = tk.Label(self, text = ' =~> ')
		self.space.grid(row=1, column=12, columnspan=2)

		self.space = tk.Label(self, text = '    ')
		self.space.grid(row=2, column=12, rowspan=2)

		self.space = tk.Label(self, text = '   ')
		self.space.grid(row=15, column=12, rowspan=2)

		self.btn1 = tk.Button(self, text='Much Better', width=20, pady=5,
			command=self.feedback1, borderwidth=2,relief="raised", bg='green')
		self.btn1.grid(row=17, column=25 )

		self.btn2 = tk.Button(self, text='Slightly Better', width=20, pady=5,
			command=self.feedback2,borderwidth=2,relief='raised',bg='#9EF703')
		self.btn2.grid(row=18, column=25 )

		self.btn3 = tk. Button(self, text='Same', width=20, pady=5, 
			command=self.feedback3, borderwidth=2, relief="raised")
		self.btn3.grid(row=19, column=25)

		self.btn4 = tk.Button(self, text='Slightly Worse', width=20, pady=5, 
			command=self.feedback4, borderwidth=2, relief="raised", bg='#E7720C')
		self.btn4.grid(row=20, column=25 )

		self.btn5 = tk.Button(self, text='Much Worse', width=20, pady=5, 
			command=self.feedback5, borderwidth=2, relief="raised", bg='red')
		self.btn5.grid(row=21, column=25 )

	def display_images(self, imd=None, imt=None):
		"""displays two images side by sid"""
		# im1 = [ImageTk.PhotoImage(Image.open(imd_name)), ImageTk.PhotoImage(Image.open(imt_name))]
		self.imdL = tk.Label(self, image=imd,borderwidth=4, relief="sunken")
		self.imdL.grid(row=4, column=0, rowspan=11, columnspan=12)

		self.imtL = tk.Label(self, image=imt,borderwidth=4, relief="sunken")
		self.imtL.grid(row=4, column=14, rowspan=11, columnspan=12)

	def feedback1(self):
		"""fetch the feedback of the user"""
		self.flag = True
		self.rate = 5

	def feedback2(self):
		"""fetch the feedback of the user"""
		self.flag = True
		self.rate = 4
		
	def feedback3(self):
		"""fetch the feedback of the user"""
		self.flag = True
		self.rate = 3
		
	def feedback4(self):
		"""fetch the feedback of the user"""
		self.flag = True
		self.rate = 2
		#gui.destroy()

	def feedback5(self):
		"""fetch the feedback of the user"""
		self.flag = True
		self.rate = 1
		

if __name__ == '__main__':
	import cv2
	import time

	gui = GUI()
	# gui.mainloop()
	im = cv2.imread('lena.jpg') #read an image here
	#convert to PIL image format
	imc = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	im_pil = Image.fromarray(imc)#reverse im_np = np.asarray(im_pil)

	#imd = Image.open('lena.png') 
	#resized = im_pil.resize((400,300), Image.ANTIALIAS)
	im = ImageTk.PhotoImage(im_pil) #(imd)
	im1 = [im,im]

	for i in range(50):
		gui.display_images(im1[0],im1[1])
		time.sleep(0.1)
		gui.update()
	
