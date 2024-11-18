#from .models import Project, Task
import csv
from io import TextIOWrapper
from io import StringIO

#Librerias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.templatetags.static import static

import base64
from io import BytesIO

#Para el modelo de optimización
from .models import OptimizationModel

# Para hacer redirect
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect


# - - - - - - - - - - - - - Procesamiento del csv - - - - - - - - - - - - - 
import datetime
import locale  #Para que de el día en español

#Esto lo hago para crear un vector de los periodos y agregarlo a la tabla
import calendar

# Create your views here.
def hello(request):
    return HttpResponse("<h1>Hello world</h1>")

def index(request):
    return render(request,"index.html")

def about(request):
    return render(request,"about.html")

def calculadora(request):
    return render(request,"calculadora.html")

def pruebas(request):
    return render(request,"pruebas.html")

def login(request):
    return render(request,"login.html")

def contact(request):
    return render(request,"contact.html")

def upload(request):
    if request.method == 'POST' and request.FILES['csvFile']:
        file = request.FILES['csvFile']

        if file.name.endswith('.csv'):
            # Procesar el archivo CSV
            titles, data, df, grafica_base64 = process_csv(file)

            # Asegúrate de imprimir o depurar los títulos y datos aquí
            print(titles)
            print(data)

            # Pasar títulos y datos al contexto de la plantilla
            return render(request, 'result.html', {'titles': titles, 'data': data, 'df':df, 'grafica_base64': grafica_base64})
            #return HttpResponseRedirect('/result'), render(request, 'result.html', {'titles': titles, 'data': data, 'df':df, 'grafica_base64': grafica_base64})
        else:
            return HttpResponse("Se debe seleccionar un archivo CSV válido.")

    return HttpResponse("No se seleccionó ningún archivo CSV.")

def process_csv(file): #Solo usa df, el resto se podria eliminar (titles y data)
    # Envolver el archivo en TextIOWrapper para asegurarse de que se abra en modo de texto
    csv_file = TextIOWrapper(file, encoding='utf-8')

    # Leer el contenido del archivo CSV con pandas
    df = pd.read_csv(csv_file, delimiter=';')

    # Obtener los títulos de las columnas
    titles = list(df.columns)

    # Obtener los datos del DataFrame
    data = df.to_dict('records')

    #Creo el eje y
    y=list(np.arange(1,35041,1))

    # Crear la gráfica de la curva de carga
    plt.figure(figsize=(10,4))  # Ancho x Alto en pulgadas
    plt.plot(y, df['Consumo (kwh)'])
    plt.xlabel("Tiempo (horas)")
    plt.xlim([1,35041])
    plt.ylabel("Consumo (kWh)")
    #plt.title('Curva de Carga')
    plt.grid()

    # Convertir la gráfica a base64 para mostrarla directamente en la plantilla
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    grafica_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    # IMAGEN PERMANENTE
    # Guardar el DataFrame modificado en un nuevo archivo CSV

    # Generar la gráfica de la curva de carga
    grafica_original = "grafica_original.png"
    generate_load_curve_graph_perm(df, filename= grafica_original)

    # - - - - - - - - - - - - - - - - - - - Procesamiento del csv - - - - - - - - - - - - - - - - - - - 

    df=pd.read_csv('Medidasprueba3.csv', sep=';')

    #Código para recorrer toda la tabla, crear un vector con los días de la semana, meses y año, y ponerlo en la tabla
    fechas_list = df['Fecha de la lectura']

    type(fechas_list)

    dias_semana = []
    meses = []
    anos = []

    for fecha in fechas_list:
        dia = int(fecha[0:2])
        mes = int(fecha[3:5])
        ano = int(fecha[6:10])
        fecha = datetime.date(ano,mes,dia)
        nombre_dia = fecha.strftime('%A')
        #print(nombre_dia)
        dias_semana.append(nombre_dia)
        meses.append(mes)
        anos.append(ano)


    #Código para recorrer toda la tabla, crear un vector con las horas y los minutos, y ponerlo en la tabla
    horas_list = df['Hora de la lectura']

    horas = []
    minutos = []

    for hora in horas_list:
        h = hora[0:2]
        m = hora[2:5]
        
        if h[1]==":":  #Eliminamos los 2 puntos cuando la hora es de 2 cifras
            h = hora[0:1]
        
        if m[0]==":":  #Eliminamos los 2 puntos cuando la hora es de 2 cifras
            m = hora[3:5]
        if m=="00": #En el caso que la hora sea en punto, pasamos a 45 min
            m = "45"
            if h=="0":  #Si es hora 0, pasamos a las 23h
                h = "23"
            else:  #Resto de casos en los que la hora no es 0, restamos 1h
                h = str(int(h)-1)
        else:  #En el caso de que la hora no sea en punto, restamos 15 minutos
            m = str(int(m)-15)

        
        horas.append(h)
        minutos.append(m)

    # Insertar la nueva columna en una ubicación específica, en este caso después de la columna 'A'
    df["Hora inicio"]=horas
    df["Minutos inicio"]=minutos
    df["Dia de la semana"]=dias_semana
    df["Mes"]=meses
    df["Año"]=anos


    # Lectura del csv con los periodos
    periodos=pd.read_csv('Periodos copy.csv', sep=';',encoding='iso-8859-1')


    #Necesito la hora, día de la semana y mes
    horas_list = df['Hora inicio']
    dias_list = df['Dia de la semana']
    meses_list = df['Mes']

    #enero, febrero, julio y diciembre
    #marzo y noviembre
    #abril, mayo y octubre
    #junio, agosto y septiembre

    periodos_list = []

    for i in range(len(df)):
        if dias_list[i]=="sábado" or dias_list[i]=="domingo": #Fines de semana (faltan agregar los festivos)
            #print(horas_list[i]+"-"+dias_list[i]+" = P6")
            periodos_list.append("P6")
        else:  #El resto de dias
            nombre_mes = calendar.month_name[meses_list[i]].capitalize()
            #print(horas_list[i]+"-"+dias_list[i]+" = "+periodos.loc[int(horas_list[i]), calendar.month_name[meses_list[i]].capitalize()])
            periodos_list.append(periodos.loc[int(horas_list[i]), nombre_mes])
        
    #Coloco el vector de periodos en la tabla
    df["Periodo"]=periodos_list

    print(type(df))

    #df.to_csv('datosProcesados.csv')
    df.to_excel('datosProcesados.xlsx', index=False)
    df.to_excel('datosProcesadosModificados.xlsx', index=False)

    return titles, data, df, grafica_base64


#Modelo de optimización simple
def modelo(request):
    # Crear una instancia del modelo de optimización
    model_instance = OptimizationModel()

    # Resolver el modelo
    model_instance.solve()

    # Pasar el resultado a la plantilla
    context = {'resultado': model_instance.objetivo,
               'pc1': model_instance.pc1,
               'pc2': model_instance.pc2,
               'pc3': model_instance.pc3,
               'pc4': model_instance.pc4,
               'pc5': model_instance.pc5,
               'pc6': model_instance.pc6,
               }
    
    # Guardar datos en un fichero
    with open('modelo.txt', 'w') as f:
        f.write(f"{model_instance.objetivo}:{model_instance.pc1}:{model_instance.pc2}:{model_instance.pc3}:{model_instance.pc4}:{model_instance.pc5}:{model_instance.pc6}\n")
    
    # Renderizar la plantilla HTML con el resultado
    return render(request, 'modelo.html', context)

#Modelo de optimización complejo
def modeloComplejo(request):
    # Crear una instancia del modelo de optimización
    model_instance = OptimizationModel()

    # Resolver el modelo
    model_instance.solveComplejo()

    # Pasar el resultado a la plantilla
    context = {'resultado': model_instance.objetivo,
               'pc1': model_instance.pc1,
               'pc2': model_instance.pc2,
               'pc3': model_instance.pc3,
               'pc4': model_instance.pc4,
               'pc5': model_instance.pc5,
               'pc6': model_instance.pc6,
               }
    
    # Renderizar la plantilla HTML con el resultado
    return render(request, 'modelo.html', context)



import json
from django.shortcuts import render
from django.http import JsonResponse

import json
from django.shortcuts import render, redirect
from django.http import JsonResponse

import pandas as pd
import numpy as np
import json
import shutil
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Mapeo de nombres de días de la semana de español a inglés
dias_esp_ing = {
    'lunes': 'Monday',
    'martes': 'Tuesday',
    'miércoles': 'Wednesday',
    'jueves': 'Thursday',
    'viernes': 'Friday',
    'sábado': 'Saturday',
    'domingo': 'Sunday'
}

def generate_load_curve_graph(df):
    # Crear la gráfica de la curva de carga
    plt.figure(figsize=(10,4))  # Ancho x Alto en pulgadas
    plt.plot(df['Consumo (kwh)'])
    plt.xlabel("Tiempo (horas)")
    plt.ylabel("Consumo (kWh)")
    plt.grid()

    # Convertir la gráfica a base64 para mostrarla directamente en la plantilla
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    grafica_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return grafica_base64

import matplotlib.pyplot as plt
import os

def generate_load_curve_graph_perm(df, filename='load_curve_graph.jpeg'):
    print("PASO 1")
    df=pd.read_csv('datosProcesadosModificados.csv', sep=',')
    print("PASO 2")
    # Crear la gráfica de la curva de carga
    plt.figure(figsize=(10, 4))  # Ancho x Alto en pulgadas
    print("PASO 3")
    plt.plot(df['Consumo (kwh)'])  # Trazar la columna 'Consumo (kwh)' del DataFrame
    print("PASO 4")
    plt.xlabel("Tiempo (horas)")  # Etiqueta del eje x
    plt.ylabel("Consumo (kWh)")  # Etiqueta del eje y
    plt.grid()  # Mostrar cuadrícula en la gráfica
    
    # Ruta donde guardar la imagen en el directorio static
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    file_path = os.path.join(static_dir, filename)
    
    # Guardar la gráfica en el disco
    plt.savefig(file_path, format='jpeg', transparent=True)


def guardar_modificacion(request):
    if request.method == "POST":
        try:
            # Recuperar los datos del formulario
            opcion_seleccionada = request.POST.get("opciones")

            datos_formulario = {
                "opcion_seleccionada": opcion_seleccionada,
            }

            # Añadir campos adicionales según la opción seleccionada
            if opcion_seleccionada in ["reducir_porcentaje", "aumentar_porcentaje"]:
                datos_formulario["porcentaje"] = int(request.POST.get("porcentaje"))
                datos_formulario["inicio"] = request.POST.get("inicioPorcentaje")
                datos_formulario["fin"] = request.POST.get("finPorcentaje")
                datos_formulario["dias"] = request.POST.getlist("diasPorcentaje")
                datos_formulario["hora_inicio"] = request.POST.get("horaInicioPorcentaje")
                datos_formulario["hora_fin"] = request.POST.get("horaFinPorcentaje")
            elif opcion_seleccionada in ["reducir_kwh", "aumentar_kwh"]:
                datos_formulario["kwh"] = int(request.POST.get("kwh"))
                datos_formulario["inicio"] = request.POST.get("inicioKwh")
                datos_formulario["fin"] = request.POST.get("finKwh")
                datos_formulario["dias"] = request.POST.getlist("diasKwh")
                datos_formulario["hora_inicio"] = request.POST.get("horaInicioKwh")
                datos_formulario["hora_fin"] = request.POST.get("horaFinKwh")
            elif opcion_seleccionada == "estimacion":
                datos_formulario["tipo_estimacion"] = request.POST.get("estimacion")
                datos_formulario["porcentaje"] = int(request.POST.get("porcentajeEstimacion"))
            elif opcion_seleccionada == "mover_demanda":
                datos_formulario["porcentaje"] = int(request.POST.get("porcentajeMoverDemanda"))
                datos_formulario["inicio_origen"] = request.POST.get("inicioMoverDemandaOrigen")
                datos_formulario["fin_origen"] = request.POST.get("finMoverDemandaOrigen")
                datos_formulario["dias_origen"] = request.POST.getlist("diasMoverDemandaOrigen")
                datos_formulario["hora_inicio_origen"] = request.POST.get("horaInicioMoverDemandaOrigen")
                datos_formulario["hora_fin_origen"] = request.POST.get("horaFinMoverDemandaOrigen")
                datos_formulario["inicio_destino"] = request.POST.get("inicioMoverDemandaDestino")
                datos_formulario["fin_destino"] = request.POST.get("finMoverDemandaDestino")
                datos_formulario["dias_destino"] = request.POST.getlist("diasMoverDemandaDestino")
                datos_formulario["hora_inicio_destino"] = request.POST.get("horaInicioMoverDemandaDestino")
                datos_formulario["hora_fin_destino"] = request.POST.get("horaFinMoverDemandaDestino")

            # Guardar los datos en el archivo JSON
            with open('modificaciones.json', 'r') as archivo:
                data = archivo.read()
                if data:
                    modificaciones = json.loads(data)
                else:
                    modificaciones = []

            modificaciones.append(datos_formulario)

            with open('modificaciones.json', 'w') as archivo:
                json.dump(modificaciones, archivo, indent=4)
                archivo.write('\n')  # Agrega un salto de línea para separar las entradas

            # Una vez que se han guardado los datos, ejecutar las operaciones en el DataFrame
            # Paso 1: Copiar el archivo datosProcesados.csv a datosProcesados_modificado.csv
            shutil.copyfile('datosProcesados.csv', 'datosProcesados_modificado.csv')

            # Paso 2: Abrir el archivo datosProcesados_modificado.csv
            df_modificado = pd.read_csv('datosProcesados_modificado.csv')

            # Paso 3: Convertir la columna de fechas al formato utilizado en modificaciones.json
            df_modificado['Fecha de la lectura'] = pd.to_datetime(df_modificado['Fecha de la lectura'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')

            # Paso 4: Abrir el archivo modificaciones.json
            with open('modificaciones.json', 'r') as file:
                modificaciones = json.load(file)

            # Paso 5: Iterar sobre cada modificación
            for modificacion in modificaciones:
                opcion_seleccionada = modificacion.get('opcion_seleccionada')
                inicio_modificacion = modificacion.get('inicio')
                fin_modificacion = modificacion.get('fin')
                dias_modificacion_esp = modificacion.get('dias')  # Obtener la lista de días de la modificación en español
                porcentaje = modificacion.get('porcentaje') / 100  # Convertir porcentaje a decimal
                valor_kwh = modificacion.get('kwh')

                # OPCION DE AUMENTAR PORCENTAJE
                if opcion_seleccionada == 'aumentar_porcentaje' and inicio_modificacion and fin_modificacion and dias_modificacion_esp:
                    # Convertir los nombres de los días de la semana de español a inglés
                    dias_modificacion_ing = [dias_esp_ing[dia] for dia in dias_modificacion_esp if dia in dias_esp_ing]

                    # Convertir inicio y fin a objetos datetime
                    inicio_modificacion = pd.to_datetime(inicio_modificacion)
                    fin_modificacion = pd.to_datetime(fin_modificacion)

                    # Filtrar el DataFrame para obtener las filas dentro del rango de fechas de la modificación y en los días especificados
                    df_filtrado = df_modificado[(df_modificado['Fecha de la lectura'] >= inicio_modificacion.strftime('%Y-%m-%d')) &
                                                (df_modificado['Fecha de la lectura'] <= fin_modificacion.strftime('%Y-%m-%d')) &
                                                (df_modificado['Dia de la semana'].isin(dias_modificacion_ing))]  # Filtrar por días traducidos

                    # Convertir la columna 'Consumo (kwh)' a tipo float
                    df_modificado['Consumo (kwh)'] = df_modificado['Consumo (kwh)'].astype(float)

                    # Aplicar el porcentaje de aumento al consumo en cada fila del DataFrame filtrado
                    df_modificado.loc[df_filtrado.index, 'Consumo (kwh)'] *= (1 + porcentaje)

                    # Convertir la columna 'Consumo (kwh)' de nuevo a tipo int64
                    df_modificado['Consumo (kwh)'] = df_modificado['Consumo (kwh)'].astype(int)

                # OPCION DE REDUCIR PORCENTAJE
                elif opcion_seleccionada == 'reducir_porcentaje' and inicio_modificacion and fin_modificacion and dias_modificacion_esp:
                    # Convertir los nombres de los días de la semana de español a inglés
                    dias_modificacion_ing = [dias_esp_ing[dia] for dia in dias_modificacion_esp if dia in dias_esp_ing]

                    # Convertir inicio y fin a objetos datetime
                    inicio_modificacion = pd.to_datetime(inicio_modificacion)
                    fin_modificacion = pd.to_datetime(fin_modificacion)

                    # Filtrar el DataFrame para obtener las filas dentro del rango de fechas de la modificación y en los días especificados
                    df_filtrado = df_modificado[(df_modificado['Fecha de la lectura'] >= inicio_modificacion.strftime('%Y-%m-%d')) &
                                                (df_modificado['Fecha de la lectura'] <= fin_modificacion.strftime('%Y-%m-%d')) &
                                                (df_modificado['Dia de la semana'].isin(dias_modificacion_ing))]  # Filtrar por días traducidos

                    # Convertir la columna 'Consumo (kwh)' a tipo float
                    df_modificado['Consumo (kwh)'] = df_modificado['Consumo (kwh)'].astype(float)

                    # Aplicar el porcentaje de aumento al consumo en cada fila del DataFrame filtrado
                    df_modificado.loc[df_filtrado.index, 'Consumo (kwh)'] *= (1 - porcentaje)

                    # Convertir la columna 'Consumo (kwh)' de nuevo a tipo int64
                    df_modificado['Consumo (kwh)'] = df_modificado['Consumo (kwh)'].astype(int)

                # OPCION DE AUMENTAR VALOR
                elif opcion_seleccionada == 'aumentar_kwh' and inicio_modificacion and fin_modificacion and dias_modificacion_esp:
                    # Convertir los nombres de los días de la semana de español a inglés
                    dias_modificacion_ing = [dias_esp_ing[dia] for dia in dias_modificacion_esp if dia in dias_esp_ing]

                    # Convertir inicio y fin a objetos datetime
                    inicio_modificacion = pd.to_datetime(inicio_modificacion)
                    fin_modificacion = pd.to_datetime(fin_modificacion)

                    # Filtrar el DataFrame para obtener las filas dentro del rango de fechas de la modificación y en los días especificados
                    df_filtrado = df_modificado[(df_modificado['Fecha de la lectura'] >= inicio_modificacion.strftime('%Y-%m-%d')) &
                                                (df_modificado['Fecha de la lectura'] <= fin_modificacion.strftime('%Y-%m-%d')) &
                                                (df_modificado['Dia de la semana'].isin(dias_modificacion_ing))]  # Filtrar por días traducidos

                    # Convertir la columna 'Consumo (kwh)' a tipo float
                    df_modificado['Consumo (kwh)'] = df_modificado['Consumo (kwh)'].astype(float)

                    # Aplicar el porcentaje de aumento al consumo en cada fila del DataFrame filtrado
                    df_modificado.loc[df_filtrado.index, 'Consumo (kwh)'] += valor_kwh

                    # Convertir la columna 'Consumo (kwh)' de nuevo a tipo int64
                    df_modificado['Consumo (kwh)'] = df_modificado['Consumo (kwh)'].astype(int)

                # OPCION DE REDUCIR VALOR
                elif opcion_seleccionada == 'reducir_kwh' and inicio_modificacion and fin_modificacion and dias_modificacion_esp:
                    # Convertir los nombres de los días de la semana de español a inglés
                    dias_modificacion_ing = [dias_esp_ing[dia] for dia in dias_modificacion_esp if dia in dias_esp_ing]

                    # Convertir inicio y fin a objetos datetime
                    inicio_modificacion = pd.to_datetime(inicio_modificacion)
                    fin_modificacion = pd.to_datetime(fin_modificacion)

                    # Filtrar el DataFrame para obtener las filas dentro del rango de fechas de la modificación y en los días especificados
                    df_filtrado = df_modificado[(df_modificado['Fecha de la lectura'] >= inicio_modificacion.strftime('%Y-%m-%d')) &
                                                (df_modificado['Fecha de la lectura'] <= fin_modificacion.strftime('%Y-%m-%d')) &
                                                (df_modificado['Dia de la semana'].isin(dias_modificacion_ing))]  # Filtrar por días traducidos

                    # Convertir la columna 'Consumo (kwh)' a tipo float
                    df_modificado['Consumo (kwh)'] = df_modificado['Consumo (kwh)'].astype(float)

                    # Aplicar el porcentaje de aumento al consumo en cada fila del DataFrame filtrado
                    df_modificado.loc[df_filtrado.index, 'Consumo (kwh)'] -= valor_kwh

                    # Convertir la columna 'Consumo (kwh)' de nuevo a tipo int64
                    df_modificado['Consumo (kwh)'] = df_modificado['Consumo (kwh)'].astype(int)


            # Guardar el DataFrame modificado en un nuevo archivo CSV
            df_modificado.to_csv('datosProcesadosModificados.csv', index=False)
            print("paso1")

            # Generar la gráfica de la curva de carga
            grafica_modificada = "grafica_modificada.png"
            df=pd.read_csv('datosProcesadosModificados.csv', sep=',')
            generate_load_curve_graph_perm(df, filename= grafica_modificada)


            try:
                # Intenta abrir el archivo JSON y cargar los datos
                with open('modificaciones.json', 'r') as archivo:
                    data = archivo.read()
                    # Si el archivo está vacío, inicializa data como una lista vacía
                    if not data:
                        data = '[]'
                    modificaciones = json.loads(data)
            except FileNotFoundError:   
                # Si el archivo no se encuentra, inicializa modificaciones como una lista vacía
                modificaciones = []

            print(modificaciones)  # Imprime los datos para verificarlos

            # Pasa las modificaciones a la plantilla para mostrar en la tabla

            # Renderizar la plantilla con la gráfica
            return render(request, 'menuModificaciones.html', {'grafica_modificada': grafica_modificada, 'datos': modificaciones})

        except Exception as e:
            print("Error:", str(e))
            return JsonResponse({"mensaje": "Error al procesar las modificaciones."}, status=500)

    else:
        return JsonResponse({"mensaje": "Error: se esperaba una solicitud POST."}, status=400)

def menu_modificaciones(request):
    try:
        # Intenta abrir el archivo JSON y cargar los datos
        with open('modificaciones.json', 'r') as archivo:
            data = archivo.read()
            # Si el archivo está vacío, inicializa data como una lista vacía
            if not data:
                data = '[]'
            modificaciones = json.loads(data)
    except FileNotFoundError:   
        # Si el archivo no se encuentra, inicializa modificaciones como una lista vacía
        modificaciones = []

    print(modificaciones)  # Imprime los datos para verificarlos

    # Pasa las modificaciones a la plantilla para mostrar en la tabla
    return render(request, 'menuModificaciones.html', {'datos': modificaciones})

def modificaciones(request):
    # Aquí va la lógica de tu vista
    return render(request, 'modificaciones.html')

def guardar_login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Guardar datos en un fichero
        with open('login_data.txt', 'w') as f:
            f.write(f"{email}:{password}\n")
    return render(request, 'index.html')

def guardar_registro(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Guardar datos en un fichero
        with open('register_data.txt', 'a') as f:
            f.write(f"{name}:{email}:{password}\n")
        
        with open('login_data.txt', 'w') as f:
            f.write(f"{email}:{password}\n")

    return render(request, 'index.html')

def perfil(request):
    try:
        with open('login_data.txt', 'r') as f:
            content = f.read().strip()
            email, password = content.split(':')
    except (FileNotFoundError, ValueError):
        email = password = ''

    context = {
        'email': email,
        'password': password,
    }

    return render(request, 'perfil.html', context)

def pdf(request):
    return render(request, 'index.html')