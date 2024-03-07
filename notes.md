# ML PROJECTS TELEMATICS

## PROBLEMS TO TACKLE
### Improve image using preprocessing need a long time
- https://medium.com/visionwizard/ improving-illumination-in-night-time-images-fac8025f4bb7
- ref from above docs, need to 1-5 minutes for processing
- Need to do 7 Steps, to preprocessing image in night time:
  ![alt text](image-1.png)
- prefer to used infrared camera
- better no need image processing
- Comparation RGB vs Infrared Camera![alt text](IRCamera_VS_RGB.png)

## To doing now
- image preprocessing just only converted to gray contrast
- night time with lamp condition after preprocessing ![alt text](image-2.png)
- night time without lamp condition after preprocessing![alt text](image-4.png)
- comparation gray scaling vs RGB good lamp cond![alt text](image-5.png)
- comparation gray scaling vs RGB notgood lamp cond![alt text](image-6.png)
- detected Face Landmarks CV2 facelandmark 68 (x, y)-coordinates![alt text](image-8.png)
- calculating eye euclidean AVG left eye & right eye![alt text](image-7.png)
- calculating mouth euclidean ![alt text](image-9.png)
- Make a condition when eyes euclidean < treshold 0.13 in 5 frame = 0.12 s (assume: camera speed 60fps)
- normal eyes![alt text](image-3.png)
- drow eyes![alt text](image-11.png)
- Make a condition when mouth euclidean > treshold 95 in 15 frame = 0.4 s (assume: camera speed 60fps) 
- normal mouth![alt text](image-12.png)
- yawn mouth![alt text](image-13.png)
- MAKING CONDITON TO JUDGMENT DROWSINESS, IF MOUTH YAWN OR EYES NEAR CLOSED, THEN DROWSINESS
- NOTES: this will not working properly in NG light conditions