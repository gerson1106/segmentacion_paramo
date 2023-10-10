import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import base64
from tensorflow import keras
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.metrics import MeanIoU
import streamlit as st
import io
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
from smooth_tiled_predictions import *
import time
CUDA_VISIBLE_DEVICES=0,1
st.set_page_config(page_icon="🌳", page_title="G&ESan-SEGMENTACIÓN", layout="wide")
logo="logo-udi-negro.png"
imagen1 = np.array(Image.open(logo))
st.image(imagen1,width=350, use_column_width=False)
st.markdown(
    """
    <style>
    .titulo-negro {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .titulo{
        color: black;
        
    }
    </style>
    """,
    unsafe_allow_html=True,
)
#text-align: center;
# Mostrar el título con el estilo aplicado
st.markdown("<h1 class='titulo-negro'>G&ESan-SEGMENTACIÓN </h1>", unsafe_allow_html=True)
st.markdown("<h1 class='titulo'>EN EL PÁRAMO DE SANTURBÁN🌳</h1>", unsafe_allow_html=True)
st.markdown("")
st.markdown("")
col4, col5 = st.columns(2)
col4, col5 = st.columns([2,2])
with col4:
 st.markdown('<p style="color: black;">El páramo de Santurbán, ubicado en Santander-Norte de Santander, es un ecosistema de gran importancia debido a su biodiversidad y su función como fuente de agua dulce para comunidades locales. Este ecosistema alberga especies endémicas y regula el flujo de agua, contribuyendo a la seguridad hídrica. Su delimitación y conservación son esenciales para preservar la biodiversidad, garantizar la calidad del agua y mitigar el cambio climático, destacando la necesidad de políticas de conservación y desarrollo sostenible en la región, la delimitación y conservación del páramo de Santurbán son esenciales para preservar esta riqueza de biodiversidad y garantizar la calidad del agua que fluye de sus tierras hacia las comunidades locales. Además, su conservación desempeña un papel vital en la mitigación del cambio climático, ya que actúa como sumidero de carbono, ayudando a reducir la concentración de gases de efecto invernadero en la atmósfera.</p>', unsafe_allow_html=True)    
with col5:
  ima="param.jpeg"
  imagen = np.array(Image.open(ima))
  st.image(imagen, caption="Laguna de las Calles", use_column_width=True)
st.markdown("<h2 style='color: black;'>¿Que es la segmentación y como funciona?</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    .texto-negro {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("""
<p class='texto-negro'>
La segmentación es un proceso en el campo de la visión por computadora que se utiliza para dividir o separar una imagen en partes significativas o regiones con características similares. Es como dividir una imagen en diferentes "pedazos" que contienen objetos o elementos que queremos identificar o analizar por separado. Este método es útil en muchas aplicaciones, como reconocimiento de objetos, medicina, procesamiento de imágenes y más, ya que permite comprender y trabajar con partes específicas de una imagen en lugar de tratarla como un conjunto completo. Funciona seleccionando características visuales y aplicando algoritmos para separar la imagen en regiones relevantes.
</p>
""", unsafe_allow_html=True)
seg = "segmen.jpeg"
imagen2 = Image.open(seg)

# Crea tres columnas
col1, col2, col3 = st.columns([1,6,1])

# Muestra la imagen en la columna del medio
with col2:
    st.image(imagen2, caption="Segmentación imagen satelital", width=800)
st.markdown("")
st.markdown("")
st.markdown("<h2 style='color: black;'>Características de la tipología segmentada</h2>", unsafe_allow_html=True)
st.markdown("")
st.markdown("")
st.markdown("""
<p class='texto-negro'>
🟩 Forestacion Alta: Para esta clasificación se llevo a cabo la etiquetación de zonas con abundante vegetación con respecto a concentraciónes de arboles y ciertas zonas con tonalidades de verde oscuro.
""", unsafe_allow_html=True)
st.markdown("""
<p class='texto-negro'>
🟨 Forestacion Baja: Esta caracterización comprende lugares con baja vegetación o vegetación muy dispersa y zonas con tonalidades de verdes claros.
</p>
""", unsafe_allow_html=True)
st.markdown("""
<p class='texto-negro'>
🟥 Edificaciones: Esta segmentación corresponde a todo el tema de urbanización lo que son tejados de casas, edificaciones, galpones y criaderos.
</p>
""", unsafe_allow_html=True)
st.markdown("""
<p class='texto-negro'>
🟦 Cultivos: Esta representación segmentada corresponde a todo lo referente a los diferentes cultivos caracteristicos de la zona como lo es la cebolla, la papa, etc.
</p>
""", unsafe_allow_html=True)
st.markdown("""
<p class='texto-negro'>
🟪 Lagunas: La representacion de la etiquetación de lagunas es referente al agua especificamente de las lagunas.
</p>
""", unsafe_allow_html=True)
st.markdown("""
<p class='texto-negro'>
⬛ Áreas no interés: La segmentación de no áreas de no interés consta de aquellas zonas que no son relevantes dentro de la imagen tales como caminos, zonas rocosas, sombras y erosiones.
</p>
""", unsafe_allow_html=True)
st.markdown("<h2 style='color: black;'>Video para adquirir imagenes satelitales</h2>", unsafe_allow_html=True)
st.markdown("")
st.markdown("")
# URL del video (o ruta local del archivo de video)

video_url = "Google_earth.mp4"  # Reemplaza con la URL o ruta de tu video
container_width = 480  # Puedes cambiar este valor según lo desees
container_height = 270  # Puedes cambiar este valor según lo desees

# Agregar el contenedor del video con el ancho y alto personalizados
video_html = f'<video width="{container_width}" height="{container_height}" controls><source src="{video_url}" type="video/mp4"></video>'
# Crea tres columnas
col1, col2, col3 = st.columns([1,3,1])
with col2:
# Mostrar el video usando HTML personalizado
 
 # Agregar el reproductor de video
 st.video(video_url)
st.markdown("<h2 style='color: black;'>Área de experimentación</h2>", unsafe_allow_html=True)
# Path del modelo preentrenado
#MODEL_PATH = "C:/Users/gerso/OneDrive/Escritorio/pajaros/API_DeepLearning-master/models/model_VGG16.h5"
c29, c30, c31 = st.columns([1, 8, 1]) # 3 columnas: 10%, 60%, 10%
#MODEL_PATH = "C:/Users/gerso/OneDrive/Escritorio/pajaros/API_DeepLearning-master/models/model_VGG16.h5"

###Suavisar la imagen 
from smooth_tiled_predictions import predict_img_with_smooth_windowing

st.markdown("")
st.markdown("")
col1, col2 = st.columns(2)
col1, col2 = st.columns([2,2])

   

with c29:
 def main():
    
  st.title("")
input_img=""
st.markdown(
    """
    <style>
    .texto-negro {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Mostrar el texto con el estilo aplicado
st.markdown('<p class="texto-negro">Carga una imagen</p>', unsafe_allow_html=True)

img_file_buffer = st.file_uploader(" ", type=["jpg"]) 

numero = st.number_input("Ingrese las hectáreas:", min_value=None, max_value=None, step=0.01, format="%.2f")
with c30:
  with col1: 
  
   if img_file_buffer is not None: 

    image_rgb = Image.open(img_file_buffer).convert('RGB')
    
    #img = cv2.imdecode(np.frombuffer(img_file_buffer.read(), np.uint8), 1)
    if image_rgb is None:
         st.error("El archivo cargado está vacío. Por favor, carga un archivo válido.")
    else:
         img_array_rgb = np.array(image_rgb)
         img = img_array_rgb[:, :, ::-1]
    
    #col1.header("img_file_buffer")
    image = np.array(Image.open(img_file_buffer))    
    st.image(image, caption="", use_column_width=False, width=550)
    #col1.image(image, use_column_width=True, channels="BGR")
    #Predict using smooth blendi
    input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

    from keras.models import load_model
    model = load_model("modelo_200_img_120_epc.hdf5", compile=False)


    #prediction_image = Image.fromarray(prediction_with_smooth_blending)
    #st.image(prediction_image, caption="Predicción", use_column_width=False,width=280)

    with c30:
      with col1:
       if st.button("Segmentar imagen"):
        mensaje_espera = st.empty()
        mensaje_espera.text("Procesando imagen... Esto puede tomar un minuto...")
        # Realiza aquí tus operaciones de procesamiento de imagen
        time.sleep(8)  # Ejemplo de espera (reemplaza con el procesamiento real)

        # Borra el mensaje de espera
        mensaje_espera.empty()

        Hectarias_total = float(numero)
        st.markdown(f'<p class="texto-negro">Hectárea total: {Hectarias_total} Hectáreas</p>', unsafe_allow_html=True)
        # size of patches
        patch_size = 256

        # Number of classes 
        n_classes = 6
        predictions_smooth = predict_img_with_smooth_windowing(
         input_img,
         window_size=patch_size,
         subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
         nb_classes=n_classes,
         pred_func=(
           lambda img_batch_subdiv: model.predict((img_batch_subdiv))
          )
         ) 
    
        final_prediction = np.argmax(predictions_smooth, axis=2)
        unique_labels, counts = np.unique(final_prediction, return_counts=True)

        # Define class names based on your classes
        class_names = ["Background", "Edificaciones", "Cultivos", "Forestacion_Alta", "Forestacion_Baja", "Laguna"]

        # Display pixel count for each class
        total_pixeles=0
        for label, count in zip(unique_labels, counts):
        #print(f"Class {class_names[label]} has {count} pixels")
          total_pixeles=count+total_pixeles

       # Function to get pixel count for a specific class
        def get_pixel_count_for_class(class_name):
         if class_name in class_names:
           index = class_names.index(class_name)
           if index in unique_labels:
              return counts[list(unique_labels).index(index)]
           return 0

        # Get pixel counts for each class
        Background_pixeles = get_pixel_count_for_class("Background")
        Edificaciones_pixeles = get_pixel_count_for_class("Edificaciones")
        Cultivos_pixeles = get_pixel_count_for_class("Cultivos")
        Forestacion_Alta_pixeles = get_pixel_count_for_class("Forestacion_Alta")
        Forestacion_Baja_pixeles = get_pixel_count_for_class("Forestacion_Baja")
        Laguna_pixeles = get_pixel_count_for_class("Laguna")

        Hectaria_por_pixel=Hectarias_total/total_pixeles

        Hectarias_Background=Hectaria_por_pixel*Background_pixeles
        Hectarias_Edificaciones=Hectaria_por_pixel*Edificaciones_pixeles
        Hectarias_Cultivos=Hectaria_por_pixel*Cultivos_pixeles
        Hectarias_Forestacion_Alta=Hectaria_por_pixel*Forestacion_Alta_pixeles
        Hectarias_Forestacion_Baja=Hectaria_por_pixel*Forestacion_Baja_pixeles
        Hectarias_Laguna=Hectaria_por_pixel*Laguna_pixeles

        def label_to_rgb(predicted_image):
    
  
         Background = '000000'.lstrip('#')
         Background = np.array(tuple(int(Background[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    
         Edificaciones = '800000'.lstrip('#')
         Edificaciones = np.array(tuple(int(Edificaciones[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
         Cultivos = '000080'.lstrip('#') 
         Cultivos = np.array(tuple(int(Cultivos[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    
         Forestacion_Alta =  '008000'.lstrip('#') 
         Forestacion_Alta = np.array(tuple(int(Forestacion_Alta[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
    
         Forestacion_Baja = '808000'.lstrip('#') 
         Forestacion_Baja = np.array(tuple(int(Forestacion_Baja[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
    
         Laguna = '800080'.lstrip('#') 
         Laguna = np.array(tuple(int(Laguna[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
    
    
    
         segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    
         segmented_img[(predicted_image == 0)] = Background
         segmented_img[(predicted_image == 1)] = Edificaciones
         segmented_img[(predicted_image == 2)] = Cultivos
         segmented_img[(predicted_image == 3)] = Forestacion_Alta
         segmented_img[(predicted_image == 4)] = Forestacion_Baja
         segmented_img[(predicted_image == 5)] = Laguna
    
    
         segmented_img = segmented_img.astype(np.uint8)
         return(segmented_img)
        mensaje_espera = st.empty()
        mensaje_espera.text("Imagen segmentada---->")
        # Realiza aquí tus operaciones de procesamiento de imagen
        time.sleep(2)  # Ejemplo de espera (reemplaza con el procesamiento real)

        # Borra el mensaje de espera
        mensaje_espera.empty()
     
        with c30:
         with col2:
          prediction_with_smooth_blending=label_to_rgb(final_prediction)
          prediction_image = Image.fromarray(prediction_with_smooth_blending)
          st.image(prediction_image,caption="", use_column_width=False,width=550)
          # Convertir la imagen PIL a bytes
          img_byte_arr = io.BytesIO()
          prediction_image.save(img_byte_arr, format='PNG')
          img_byte_arr = img_byte_arr.getvalue()
 
          # Crear un botón de descarga para la imagen
          st.download_button(
           label="Descargar Imagen Segmentada",
           data=img_byte_arr,
           file_name='imagen_segmentada.png',
           mime='image/png',
          )
          with col2:

            Hectarias_Background = round(Hectarias_Background, 2)
            Hectarias_Edificaciones = round(Hectarias_Edificaciones, 2)
            Hectarias_Cultivos = round(Hectarias_Cultivos, 2)
            Hectarias_Forestacion_Alta = round(Hectarias_Forestacion_Alta, 2)
            Hectarias_Forestacion_Baja = round(Hectarias_Forestacion_Baja, 2)
            Hectarias_Laguna = round(Hectarias_Laguna, 2)

           #st.write('<span style="font-size: 24px;">🟢 Forestacion Alta</span>', unsafe_allow_html=True)
            st.markdown(f'<p class="texto-negro">🟢 Forestacion Alta: {Hectarias_Forestacion_Alta} Hectáreas</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="texto-negro">🟡 Forestacion Baja: {Hectarias_Forestacion_Baja} Hectáreas</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="texto-negro">🔴 Edificaciones: {Hectarias_Edificaciones} Hectáreas</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="texto-negro">🔵 Cultivos: {Hectarias_Cultivos} Hectáreas</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="texto-negro">🟣 Lagunas: {Hectarias_Laguna} Hectáreas</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="texto-negro">⚫ Áreas no interés: {Hectarias_Background} Hectáreas</p>', unsafe_allow_html=True)
            #st.write('🟢 Forestacion Alta')
            #st.write('🟡 Forestacion Baja')
            #st.write('🔴 Edificaciones')
            #st.write('🔵 Cultivos')
            #st.write('🟣 Lagunas')
 
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("")
with st.expander("ℹ️-Contacto"):
 st.markdown(
    """
    <style>
    .streamlit-text-container {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
 st.markdown('<p class="texto-negro">Gerson Hernandez-Erick Capacho</p>', unsafe_allow_html=True)
 st.write("ghernandez6@udi.edu.co")
 st.write("ecapacho1@udi.edu.co")
 st.markdown("---")
if __name__ == '__main__':
 main()


