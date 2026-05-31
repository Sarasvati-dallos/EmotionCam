[app]
title = EmotionCam
package.name = emotioncam
package.domain = org.sarasvati

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,h5,xml

version = 1.0

requirements = python3,kivy==2.3.0,opencv,numpy,pillow,tensorflow-lite

orientation = portrait
fullscreen = 0

android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE
android.api = 33
android.minapi = 21
android.sdk = 33
android.ndk = 25b
android.archs = arm64-v8a

android.allow_backup = True
android.camera = True

[buildozer]
log_level = 2
warn_on_root = 1
