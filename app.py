# from imp import load_module
# import streamlit as st
# import os
# import json
# import librosa
# import numpy as np
# import librosa
# import pickle as pk
# # Preprocessing method
# from keras.models import Sequential
# from tensorflow import keras



# def make_prediction2(model, X):

#     genre_dict = {
#         0 : "blues",
#         1 : "classical",
#         2 : "country",
#         3 : "disco",
#         4 : "hiphop",
#         5 : "jazz",
#         6 : "metal",
#         7 : "pop",
#         8 : "reggae",
#         9 : "rock",
#         }

#     predictions = model.predict(X)
#     print(predictions)
#     genre = np.argmax(predictions)
#     return genre

# def preprocess_audio(file_path):
#     try:
#         # Load audio file
#         song, sr = librosa.load(file_path, duration=30)

#         # Generate 13-band MFCC features
#         mfcc = librosa.feature.mfcc(y=song, sr=sr, n_mfcc=13)
#         mfcc = (mfcc.T)
#         # reshaped_input = mfcc.reshape(mfcc.shape[0], 1249, -1)

#         return np.array(mfcc)
#     except Exception as e:
#         st.error(f"Error during preprocessing: {e}")
#         return None
    
    
# def predict(mfcc_features):
#     loaded_model = None
    
#     loaded_model = keras.models.load_model('Final_model.h5')

#     # Convert the MFCC features into a torch tensor
#     input_data = mfcc_features  # Transpose the matrix to match the expected shape

#     prediction = make_prediction2(loaded_model,input_data)
#     return prediction

# # Streamlit app
# def main():
#     st.title("Audio Genre Classification")

#     # File uploader
#     uploaded_file = st.file_uploader("Upload an audio file (.wav):", type=["wav"])

#     if st.button:

#         # Classify the genre if a file is uploaded
#         if uploaded_file is not None:
#             # Save the uploaded file temporarily
#             temp_path = "temp.wav"
#             with open(temp_path, "wb") as f:
#                 f.write(uploaded_file.read())

#             # Preprocess the audio
#             mfcc_features = preprocess_audio(temp_path)

#             # Display the processed features
#             if mfcc_features is not None:
#                 prediction = predict(mfcc_features)
#                 st.success("Audio file preprocessed successfully!")
#                 st.json({"mfcc_features": mfcc_features})
#                 st.success(f"Predicted Genre: {prediction}")

#         # Show a help message if no file is uploaded
#         else:
#             st.info("Please upload an audio file to preprocess and classify its genre.")

# if __name__ == "__main__":
#     main()
import streamlit as st
import os
import json
import librosa
import numpy as np
import librosa
from tensorflow.keras.models import load_model

def make_prediction(model, X):
    genre_dict = {
        0: "blues",
        1: "classical",
        2: "country",
        3: "disco",
        4: "hiphop",
        5: "jazz",
        6: "metal",
        7: "pop",
        8: "reggae",
        9: "rock",
    }

    predictions = model.predict(X)
    genre = np.argmax(predictions)
    return genre_dict[genre]

def preprocess_audio(file_path):
    try:
        # Load audio file
        song, sr = librosa.load(file_path, duration=30)

        # Generate 13-band MFCC features
        mfcc = librosa.feature.mfcc(y=song, sr=sr, n_mfcc=13)
        mfcc = mfcc.T  # Transpose the matrix to match the expected shape

        # Ensure the correct number of frames
        expected_frames = 1249
        if len(mfcc) < expected_frames:
            # Pad with zeros if less frames
            mfcc = np.pad(mfcc, ((0, expected_frames - len(mfcc)), (0, 0)))
        elif len(mfcc) > expected_frames:
            # Trim if more frames
            mfcc = mfcc[:expected_frames, :]

        return np.array(mfcc)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

def predict(mfcc_features):
    loaded_model = None
    try:
        # Load the pre-trained Keras model
        loaded_model = load_model('Final_model.h5')

        # Convert the MFCC features into a numpy array
        input_data = np.expand_dims(mfcc_features, axis=0)

        prediction = make_prediction(loaded_model, input_data)
        return prediction

    except Exception as e:
        st.error(f"Error loading or using the model: {e}")
        return None

# Streamlit app
def main():
    st.title("Audio Genre Classification")

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file (.wav):", type=["wav"])

    if st.button("Classify Genre"):

        # Classify the genre if a file is uploaded
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            temp_path = "temp.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            # Preprocess the audio
            mfcc_features = preprocess_audio(temp_path)

            # Display the processed features
            if mfcc_features is not None:
                prediction = predict(mfcc_features)
                if prediction is not None:
                    st.success(f"Predicted Genre: {prediction}")

        # Show a help message if no file is uploaded
        else:
            st.info("Please upload an audio file to preprocess and classify its genre.")

if __name__ == "__main__":
    main()
