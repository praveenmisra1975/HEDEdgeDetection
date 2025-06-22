import cv2
import numpy as np

from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.utils import platform
from kivy.clock import Clock
from kivy.logger import Logger
from applayout import AppLayout
from android_permissions import AndroidPermissions
from jnius import autoclass, cast
from edgedetect import EdgeDetect

#! [CropLayer]
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]
#! [CropLayer]


if platform == 'android':
    from jnius import autoclass
    from android.runnable import run_on_ui_thread
    from android import mActivity
    View = autoclass('android.view.View')

    @run_on_ui_thread
    def hide_landscape_status_bar(instance, width, height):
        # width,height gives false layout events, on pinch/spread 
        # so use Window.width and Window.height
        if Window.width > Window.height: 
            # Hide status bar
            option = View.SYSTEM_UI_FLAG_FULLSCREEN
        else:
            # Show status bar 
            option = View.SYSTEM_UI_FLAG_VISIBLE
        mActivity.getWindow().getDecorView().setSystemUiVisibility(option)
elif platform != 'ios':
    # Dispose of that nasty red dot, required for gestures4kivy.
    from kivy.config import Config 
    Config.set('input', 'mouse', 'mouse, disable_multitouch')

class MyApp(App):
    
    def build(self):
        self.layout = AppLayout()
 
       
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        context = cast('android.content.Context', PythonActivity.mActivity)
       
        asset_manager = cast('android.content.res.AssetManager', context.getAssets())
        try:
         with asset_manager.open('deploy.prototxt',2) as f1:
          
          size = f1.available()
          prototxt_buffer = bytearray(size)
          bytesRead1 = f1.read(prototxt_buffer)
          
          
         with asset_manager.open('hed_pretrained_bsds.caffemodel',2) as f2:
          size = f2.available()
          caffemodel_buffer = bytearray(size)
          bytesRead2 = f2.read(caffemodel_buffer)
          
         
         self.net = cv2.dnn.readNetFromCaffe(prototxt_buffer, caffemodel_buffer)
         #[Register]
         cv2.dnn_registerLayer('Crop', CropLayer)
         # [Register]
         Logger.info(f"Model: Model has loaded successfully ({self.net})")
         EdgeDetect.class_net_attribute=self.net

        except Exception as e:
         print(f"Error opening or reading file: {e}")  

      
        if platform == 'android':
            Window.bind(on_resize=hide_landscape_status_bar)
        return self.layout

    def on_start(self):
        self.dont_gc = AndroidPermissions(self.start_app)

    def start_app(self):
        self.dont_gc = None
        # Can't connect camera till after on_start()
        Clock.schedule_once(self.connect_camera)

    def connect_camera(self,dt):
        self.layout.edge_detect.connect_camera(analyze_pixels_resolution = 720,
                                               enable_analyze_pixels = True,
                                               enable_video = False)

    def on_stop(self):
        self.layout.edge_detect.disconnect_camera()

MyApp().run()

