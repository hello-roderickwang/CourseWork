#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-03-02 08:41:27
# @Author  : Xuenan(Roderick) Wang
# @email   : roderick_wang@outlook.com
# @Link    : https://github.com/hello-roderickwang


class App:
	def __init__(self, num):
		print('num in SuperClass is ',num)

	def start(self):
		print('starting')

class Android(App):
	def __init__(self, num):
		self.App(5)
		print('num in SubClass is ', num)

	def getVersion(self):
		print('Android version')

# app = Android()
# app.start()
# app.getVersion()
app = Android(10)
app.start()
app.getVersion()