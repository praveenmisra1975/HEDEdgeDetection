from kivy.clock import mainthread
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
import numpy as np
import cv2
import docdetect
from camera4kivy import Preview
from kivy.logger import Logger



class EdgeDetect(Preview):
    class_net_attribute = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analyzed_texture = None
   

    ####################################
    # Analyze a Frame - NOT on UI Thread
    ####################################

    def analyze_pixels_callback(self, pixels, image_size, image_pos, scale, mirror):
        # pixels : analyze pixels (bytes)
        # image_size   : analyze pixels size (w,h)
        # image_pos    : location of Texture in Preview (due to letterbox)
        # scale  : scale from Analysis resolution to Preview resolution
        # mirror : true if Preview is mirrored
        
        #rgba   = np.fromstring(pixels, np.uint8).reshape(image_size[1],
        #                                                 image_size[0], 4)


        frame   = np.fromstring(pixels, np.uint8).reshape(image_size[1],
                                                         image_size[0], 4)
        (H, W) = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        #hed=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        
        
        
        blob =cv2.dnn.blobFromImage(rgb, scalefactor=1.0, size=(100, 150),mean=(104, 116, 122),swapRB=False, crop=False)
        
        (EdgeDetect.class_net_attribute).setInput(blob)
        hed = (EdgeDetect.class_net_attribute).forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype("uint8")

        kernel_size = 3  
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        hed = cv2.erode(hed, kernel,iterations = 5)
        
        hed = cv2.adaptiveThreshold(hed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,27,-6)
        
        ekernel_size = 1  
        ekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ekernel_size, ekernel_size))
        hed = cv2.dilate(hed, ekernel,iterations = 3)
       


        hed=cv2.cvtColor(hed,cv2.COLOR_GRAY2RGB)
        
        rects1 = docdetect.process(hed,7,100,200)
        rects2 = docdetect.process(hed,7,90,200)
        rects3 = docdetect.process(hed,7,30,200)
        
        

        rects=rects1+rects2+rects3
       
        rgb = docdetect.draw(rects, rgb )
        
        rgba   = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA) 
       
        pixels = rgba.tostring()

        self.make_thread_safe(pixels, image_size) 

    @mainthread
    def make_thread_safe(self, pixels, size):
        if not self.analyzed_texture or\
           self.analyzed_texture.size[0] != size[0] or\
           self.analyzed_texture.size[1] != size[1]:
            self.analyzed_texture = Texture.create(size=size, colorfmt='rgba')
            self.analyzed_texture.flip_vertical()
        if self.camera_connected:
            self.analyzed_texture.blit_buffer(pixels, colorfmt='rgba') 
        else:
            # Clear local state so no thread related ghosts on re-connect
            self.analyzed_texture = None
            
    ################################
    # Annotate Screen - on UI Thread
    ################################

    def canvas_instructions_callback(self, texture, tex_size, tex_pos):
        # texture : preview Texture
        # size    : preview Texture size (w,h)
        # pos     : location of Texture in Preview Widget (letterbox)
        # Add the analyzed image
        if self.analyzed_texture:
            Color(1,1,1,1)
            Rectangle(texture= self.analyzed_texture,
                      size = tex_size, pos = tex_pos)
            


           










