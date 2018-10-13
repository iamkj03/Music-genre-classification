# Download AU files
If you want to see the dataset, follow these steps. However, you can't use au files because our code is set for reading images that are preprocessed from audio files. But you can know how we preprocessed the audio files by reading our code.
1. Download GTZAN set in http://marsyasweb.appspot.com/download/data_sets/
2. Click "Download the GTZAN genre collection"

# How to run the model(xception)
1. Comment from line 168 to the end of the file
2. Create a folder named "saved_models" on the same level of xception.py
3. Run 
```python3 xception.py```
4. After training, uncomment the comment block and comment from line 148 to line 165
5. In line 168, change the h5 file name into the file that had the highest training accuracy 
6. Run 
```python3 xception.py```.
Then you will get the AUC and the Average Precision

- Worked as a team project with Doeun Kim (https://github.com/97dawn/music-genre-classification)
