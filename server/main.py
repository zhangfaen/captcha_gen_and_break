# -*- coding: utf-8 -*-
# !/usr/bin/env python

import os
from keras.models import model_from_json
import keras.backend as K
import logging
import traceback
import web
import random
import log
from PIL import Image
import time
import numpy as np


# stop the autoreload mode of web.py
web.config.debug = False
urls = (
    '/', 'Index',
)
class Index:
    def __init__(self):
        self.ret = dict()
        self.ret["code"] = 0
        self.ret["data"] = ""
        self.res= open('res.html').read()
        self.res2='''</td></tr></tbody></table></body></html>'''
        print 'init'
       
    def GET(self):
        web.header("Content-Type","text/html; charset=utf-8")
        return self.res+self.res2
    def POST(self):
        res=''
        try:
            x = web.input(file0={})
            content = x.file0.file.read()
            if(len(content)>50):
                filename= 'static/0.png'
                os.remove(filename)
                fout = open(filename,'w') 
                fout.write(content)
                fout.close()
                img = Image.open(filename)
                print np.asarray(img).shape
                data = img.resize([160 ,60])
                data = np.multiply(data, 1 / 255.0)
                data = np.asarray(data)
                mf=open('cnn_matrix.json')
                model = model_from_json(mf.read())
                mf.close()
                model.load_weights('cnn_matrix.h5')
                preds = model.predict(np.reshape(data,(1,60, 160, 3)))
                K.clear_session()
                pred = preds[0]
                num = str(pred[0].argmax()) + str(pred[1].argmax()) + str(pred[2].argmax()) + str(pred[3].argmax())
                res = self.res+num+self.res2
                logging.info("/ responses successfully")
            else:
                res = self.res+self.res2
        except Exception as e:
            print '[DEBUG] there is an exception:', e
            self.ret["code"] = -1001
            self.ret["data"] = str(e)
            exstr = traceback.format_exc()
            print exstr 
        finally:
            return res

if __name__ == '__main__':
    log.init_log("./log", level=logging.DEBUG)
    app = web.application(urls, globals())
    app.run()
