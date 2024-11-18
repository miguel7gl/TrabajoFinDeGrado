from django.db import models
from pyomo.environ import *
import os

import numpy as np

from pyomo.environ import *
from pyomo.opt import SolverFactory

import pandas as pd

class OptimizationModel(models.Model):
    #objetivo = models.FloatField()
    #pc1 = models.FloatField()
    #pc2 = models.FloatField()
    #pc3 = models.FloatField()
    #pc4 = models.FloatField()
    #pc5 = models.FloatField()
    #pc6 = models.FloatField()

    def solve(self):

        df = pd.read_excel('datosProcesadosModificados.xlsx')
        
        # Definir el diccionario para almacenar las listas
        consumo_por_periodo_mes_dia = {}

        # Iterar sobre cada fila del DataFrame
        for indice, fila in df.iterrows():
            periodo = fila['Periodo']
            mes = fila['Mes']
            dia = fila['Dia de la semana']
            consumo = fila['Consumo (kwh)']

            # Crear la clave combinada de 'Periodo', 'Mes' y 'Dia'
            clave = f"Potencia_{periodo}_{mes}_{dia}"

            # Si la clave aún no está en el diccionario, crear una lista vacía para almacenar los consumos
            if clave not in consumo_por_periodo_mes_dia:
                consumo_por_periodo_mes_dia[clave] = []

            # Agregar el consumo a la lista correspondiente
            consumo_por_periodo_mes_dia[clave].append(consumo)

        # Calcular el promedio dentro de cada lista
        for lista_consumos in consumo_por_periodo_mes_dia.values():
            #promedio = sum(lista_consumos) / len(lista_consumos)   #PARA HACER LA MEDIA DE CADA DIA
            promedio = max(lista_consumos)                          #PARA COGER EL MAX DE CADA DIA
            # Reemplazar la lista original por el promedio
            lista_consumos.clear()
            lista_consumos.append(promedio)

        # Ahora, consumo_por_periodo_mes_dia contendrá los promedios correspondientes a cada combinación de 'Periodo', 'Mes' y 'Dia'
        consumo_por_periodo_mes_dia

        #--------------------
        #--------------------
        #--------------------

        # Diccionario para almacenar las listas agrupadas
        agrupado_por_periodo_mes = {}

        # Iterar sobre cada elemento del diccionario original
        for key, value in consumo_por_periodo_mes_dia.items():
            # Obtener el periodo y el mes del nombre de la clave
            periodo_mes = key.split("_")[1] + "_" + key.split("_")[2]

            # Comprobar si el periodo_mes ya existe en el diccionario agrupado_por_periodo_mes
            if periodo_mes not in agrupado_por_periodo_mes:
                # Si no existe, crear una nueva entrada en el diccionario con la lista correspondiente
                agrupado_por_periodo_mes[periodo_mes] = value
            else:
                # Si ya existe, añadir los valores a la lista existente
                agrupado_por_periodo_mes[periodo_mes].extend(value)

        # Imprimir el diccionario agrupado_por_periodo_mes
        for periodo_mes, valores in agrupado_por_periodo_mes.items():
            print(periodo_mes + ": ", valores)


        #--------------------
        #--------------------
        #--------------------

        # Crear una matriz de ceros con 12 filas y 6 columnas
        matriz_maximos = np.zeros((12, 6))

        # Iterar sobre los máximos para cada periodo y mes
        for periodo_mes, valores in agrupado_por_periodo_mes.items():
            maximo = max(valores)
            periodo, mes = periodo_mes.split('_')
            # Convertir el mes a un índice (restando 1 ya que los índices comienzan en 0)
            mes_idx = int(mes) - 1
            # Obtener el índice de la columna según el periodo
            if periodo == 'P1':
                col_idx = 0
            elif periodo == 'P2':
                col_idx = 1
            elif periodo == 'P3':
                col_idx = 2
            elif periodo == 'P4':
                col_idx = 3
            elif periodo == 'P5':
                col_idx = 4
            elif periodo == 'P6':
                col_idx = 5
            # Agregar el máximo a la matriz en la fila correspondiente al mes y la columna correspondiente al periodo
            matriz_maximos[mes_idx, col_idx] = maximo

        #Vectores maximetro mensual
        max_P1 = matriz_maximos[:, 0]
        max_P2 = matriz_maximos[:, 1]
        max_P3 = matriz_maximos[:, 2]
        max_P4 = matriz_maximos[:, 3]
        max_P5 = matriz_maximos[:, 4]
        max_P6 = matriz_maximos[:, 5]

        # Vector dias
        dias = [31.0, 29.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0]



        # Datos del problema 6.1 TD
        Tp_p_values = [20.557850, 12.762884, 9.926251, 7.848380, 0.325141, 0.325141]  # euros/kW año
        k_p_values = [1.000000, 0.620828, 0.482845, 0.381770, 0.015816, 0.015816]
        tep_value = 3.566788  # euros/kW

        ######################
        Tp_p_values = [15.713047, 9.547036, 4.658211, 4.142560, 2.285209, 1.553638]  # euros/kW año
        k_p_values = [1.000000, 0.620828, 0.482845, 0.381770, 0.015816, 0.015816]
        tep_value = 0.111643  # euros/kW
        ######################


        Tp_p_values_np = np.array(Tp_p_values)
        k_p_values_np = np.array(k_p_values)

        # Crear un diccionario vacío
        Pd_pj_values = {
            1: [],
            2: [],
            3: [],
            4: [],
            5: [],
            6: []
        }

        # Iterar sobre cada período y agregar los consumos correspondientes al diccionario
        for periodo, grupo in df.head(35041).groupby('Periodo'): #Modificar el valor para elegir numero de datos
            # Extraer el número del período eliminando la letra 'P'
            numero_periodo = int(periodo[1:])
            Pd_pj_values[numero_periodo] = list(grupo['Consumo (kwh)'])

        #print(Pd_pj_values)

        # Inicializar valores máximo y mínimo con los extremos de los posibles valores
        max_value = float('-inf')
        min_value = float('inf')

        # Iterar sobre el diccionario
        for key, value in Pd_pj_values.items():
            # Verificar si la lista no está vacía antes de calcular el máximo y mínimo
            if value:
                # Encontrar el valor máximo y mínimo dentro de cada lista
                max_list_value = max(value)
                min_list_value = min(value)

                # Actualizar los valores máximo y mínimo globales
                max_value = max(max_value, max_list_value)
                min_value = min(min_value, min_list_value)

        # Imprimir los resultados solo si hay elementos en el diccionario
        if max_value != float('-inf') and min_value != float('inf'):
            print("Valor máximo en el diccionario:", max_value)
            print("Valor mínimo en el diccionario:", min_value)
        else:
            print("El diccionario está vacío o todas las listas están vacías.")


        # Inicializamos un vector para almacenar los máximos
        Pd_pj_max = []

        for clave, valores in Pd_pj_values.items():
            if valores:  # Verificar si la lista de valores no está vacía
                maximo = max(valores)
                Pd_pj_max.append(maximo)
            else:
                print(f"Advertencia: La clave {clave} no tiene valores asociados.")
                Pd_pj_max.append(0)

        # Imprimimos el vector de máximos
        #print("Vector de máximos pd_pj_max:", Pd_pj_max)
                

        #FORMA PARA LOS TIPOS DE MEDIDA 4 Y 5 (SIN LA RAIZ CUADRADA DE LOS EXCESOS)

        # Crear el modelo
        model = ConcreteModel()

        # Índices
        model.p = Set(initialize=range(len(Tp_p_values)))
        model.m = Set(initialize=range(len(dias)))

        # Introducir una variable adicional para representar la expresión max(0, x)
        model.max_term1 = Var(model.m, within=NonNegativeReals)
        model.max_term2 = Var(model.m, within=NonNegativeReals)
        model.max_term3 = Var(model.m, within=NonNegativeReals)
        model.max_term4 = Var(model.m, within=NonNegativeReals)
        model.max_term5 = Var(model.m, within=NonNegativeReals)
        model.max_term6 = Var(model.m, within=NonNegativeReals)

        # Variables de decisión
        model.Pc_p = Var(model.p, within=NonNegativeReals, bounds=(0,Pd_pj_max[0]*10))
        Pc_p_optimo = [0,0,0,0,0,0]

        epsilon = 1e-10  # Ajusta este valor según sea necesario

        # Función objetivo
        def objetivo_rule(model):

            # Potencia contratada
            sum_terms = sum(Tp_p_values[p] * model.Pc_p[p] for p in model.p)

            # Exceso de potencia
            sum_terms += sum(model.max_term1[m] * 2 * tep_value * dias[m] for m in model.m)
            sum_terms += sum(model.max_term2[m] * 2 * tep_value * dias[m] for m in model.m)
            sum_terms += sum(model.max_term3[m] * 2 * tep_value * dias[m] for m in model.m)
            sum_terms += sum(model.max_term4[m] * 2 * tep_value * dias[m] for m in model.m)
            sum_terms += sum(model.max_term5[m] * 2 * tep_value * dias[m] for m in model.m)
            sum_terms += sum(model.max_term6[m] * 2 * tep_value * dias[m] for m in model.m)

            return sum_terms

        model.objetivo = Objective(rule=objetivo_rule, sense=minimize)


        # Restricciones
        def restriccion_1_rule(model, p):
            # Pc_p <= Pc_p+1 para p en el rango de 1 a len(Tp_p_values)-1
            if p < len(Tp_p_values) - 1:
                return model.Pc_p[p] <= model.Pc_p[p + 1]
            else:
                return Constraint.Skip

        def restriccion_2_rule(model, p):
            # Pc_p >= 0 para todos los p
            return model.Pc_p[p] >= 0

        def restriccion_3_rule1(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term1[m] >= 0

        def restriccion_3_rule2(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term2[m] >= 0

        def restriccion_3_rule3(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term3[m] >= 0

        def restriccion_3_rule4(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term4[m] >= 0

        def restriccion_3_rule5(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term5[m] >= 0

        def restriccion_3_rule6(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term6[m] >= 0

        def restriccion_4_rule1(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term1[m] >= max_P1[m] - model.Pc_p[0]

        def restriccion_4_rule2(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term2[m] >= max_P2[m] - model.Pc_p[1]

        def restriccion_4_rule3(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term3[m] >= max_P3[m] - model.Pc_p[2]

        def restriccion_4_rule4(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term4[m] >= max_P4[m] - model.Pc_p[3]

        def restriccion_4_rule5(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term5[m] >= max_P5[m] - model.Pc_p[4]

        def restriccion_4_rule6(model, m):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term6[m] >= max_P6[m] - model.Pc_p[5]


        # Agregar restricciones al modelo
        model.restriccion_1 = Constraint(model.p, rule=restriccion_1_rule)
        model.restriccion_2 = Constraint(model.p, rule=restriccion_2_rule)
        model.restriccion_31 = Constraint(model.m, rule=restriccion_3_rule1)
        model.restriccion_32 = Constraint(model.m, rule=restriccion_3_rule2)
        model.restriccion_33 = Constraint(model.m, rule=restriccion_3_rule3)
        model.restriccion_34 = Constraint(model.m, rule=restriccion_3_rule4)
        model.restriccion_35 = Constraint(model.m, rule=restriccion_3_rule5)
        model.restriccion_36 = Constraint(model.m, rule=restriccion_3_rule6)
        model.restriccion_41 = Constraint(model.m, rule=restriccion_4_rule1)
        model.restriccion_42 = Constraint(model.m, rule=restriccion_4_rule2)
        model.restriccion_43 = Constraint(model.m, rule=restriccion_4_rule3)
        model.restriccion_44 = Constraint(model.m, rule=restriccion_4_rule4)
        model.restriccion_45 = Constraint(model.m, rule=restriccion_4_rule5)
        model.restriccion_46 = Constraint(model.m, rule=restriccion_4_rule6)

        
        ##########################################################################
        # Obtener la ruta al directorio actual (donde se encuentra models.py)
        current_directory = os.path.dirname(os.path.abspath(__file__))

        
        # Construir la ruta al ejecutable de GLPK
        glpk_executable = os.path.join(current_directory, '..\env\Lib\site-packages\winglpk-4.65\glpk-4.65\w64\glpsol.exe')
        
        # Construir la ruta al ejecutable de ipopt
        ipopt_executable = os.path.join(current_directory, '..\env\Lib\site-packages\Ipopt-3.14.13-win64-msvs2019-md\Ipopt-3.14.13-win64-msvs2019-md\bin\ipopt.exe')
        

        #print(glpk_executable)

        # Utilizar la ruta construida con SolverFactory
        #solver = SolverFactory('glpk', executable=glpk_executable)
        #solver = SolverFactory('ipopt', executable=ipopt_executable)

        #solver.solve(model)

        #################################################################

        # Resolver el modelo y mostrar el log
        #solver = SolverFactory('ipopt')
        solver = SolverFactory('ipopt')

        results = solver.solve(model,tee=True)

        # Imprimir los valores óptimos de Pc_p
        print("Valores óptimos de Pc_p:")
        for p in model.p:
            Pc_p_optimo[p] = model.Pc_p[p].value
            print(f"Pc_p[{p}] = {model.Pc_p[p].value} --> {round(model.Pc_p[p].value, 1)}")

        # Imprimir el valor óptimo de la función objetivo
        print("Valor óptimo de la función objetivo:", model.objetivo())
        print("Valor óptimo redondeado de la función objetivo:", round(model.objetivo(), 1))

        objetivo=model.objetivo()

        #P1 manda

        print("Restricciones activas:")
        for c in model.component_data_objects(Constraint, active=True):
            print(f"{c.name}: {c()}")

        #################################################################

        # Guardar el resultado en el objeto del modelo Django
        self.objetivo = objetivo
        self.pc1 = model.Pc_p[0].value
        self.pc2 = model.Pc_p[1].value
        self.pc3 = model.Pc_p[2].value
        self.pc4 = model.Pc_p[3].value
        self.pc5 = model.Pc_p[4].value
        self.pc6 = model.Pc_p[5].value
        self.save()

    def solveComplejo(self):

        df = pd.read_excel('datosProcesadosModificados.xlsx')
        
        # Datos del problema 6.1 TD
        Tp_p_values = [24.414407, 14.692911, 11.328635, 9.250764, 1.727525, 0.967900]  # euros/kW año
        k_p_values = [1.000000, 0.620828, 0.482845, 0.381770, 0.015816, 0.015816]
        tep_value = 3.566788  # euros/kW

        ######################
        #Tp_p_values = [15.713047, 9.547036, 4.658211, 4.142560, 2.285209, 1.553638]  # euros/kW año
        #k_p_values = [1.000000, 0.620828, 0.482845, 0.381770, 0.015816, 0.015816]
        #tep_value = 0.111643  # euros/kW
        ######################


        ######################
        #Tp_p_values = [15.713047, 9.547036, 4.658211, 4.142560, 2.285209, 1.553638]  # euros/kW año
        #k_p_values = [1.000000, 0.620828, 0.482845, 0.381770, 0.015816, 0.015816]
        #tep_value = 0.111643  # euros/kW
        ######################


        Tp_p_values_np = np.array(Tp_p_values)
        k_p_values_np = np.array(k_p_values)

        # Crear un diccionario vacío
        Pd_pj_values = {
            "1_1": [], "1_2": [], "1_3": [], "1_4": [], "1_5": [], "1_6": [], "1_7": [], "1_8": [], "1_9": [], "1_10": [], "1_11": [], "1_12": [],
            "2_1": [], "2_2": [], "2_3": [], "2_4": [], "2_5": [], "2_6": [], "2_7": [], "2_8": [], "2_9": [], "2_10": [], "2_11": [], "2_12": [],
            "3_1": [], "3_2": [], "3_3": [], "3_4": [], "3_5": [], "3_6": [], "3_7": [], "3_8": [], "3_9": [], "3_10": [], "3_11": [], "3_12": [],
            "4_1": [], "4_2": [], "4_3": [], "4_4": [], "4_5": [], "4_6": [], "4_7": [], "4_8": [], "4_9": [], "4_10": [], "4_11": [], "4_12": [],
            "5_1": [], "5_2": [], "5_3": [], "5_4": [], "5_5": [], "5_6": [], "5_7": [], "5_8": [], "5_9": [], "5_10": [], "5_11": [], "5_12": [],
            "6_1": [], "6_2": [], "6_3": [], "6_4": [], "6_5": [], "6_6": [], "6_7": [], "6_8": [], "6_9": [], "6_10": [], "6_11": [], "6_12": []
        }



        # 1 mes --> 2977 - 5.1s - Pc_p_optimo = [44.2, 44.2, 44.2, 44.2, 44.2, 44.2]
        # 2 meses --> 5665 - 1min 22s - Pc_p_optimo = [48.8, 48.8, 48.8, 48.8, 48.8, 48.8]
        # 3 meses --> 8637 - 10min 44s - Pc_p_optimo = [48.8, 48.8, 48.8, 48.8, 48.8, 48.8]
        # 4 meses --> 11517  -  4086 MB
        # 5 meses --> 14493  -  9262 MB - 4min28s en empezar
        # 6 meses --> 17373  -   MB

        

        # Iterar sobre cada fila del DataFrame
        for index, fila in df.iterrows():
            # Extraer el número del período y el mes de la fila
            numero_periodo = int(fila['Periodo'][1:])
            mes = fila['Mes']

            # Construir la clave con el formato deseado
            clave = f"{numero_periodo}_{mes}"

            # Obtener el consumo de la fila
            consumo = fila['Consumo (kwh)']

            # Verificar si la clave ya existe en el diccionario
            if clave not in Pd_pj_values:
                # Si no existe, creamos una nueva entrada con una lista vacía como valor
                Pd_pj_values[clave] = []

            # Agregar el consumo a la lista de valores asociados con esta clave
            Pd_pj_values[clave].append(consumo)


        # Ahora Pd_pj_values contiene todas las filas agrupadas según el número de período y mes


        print(Pd_pj_values["1_1"])

        # Inicializar valores máximo y mínimo con los extremos de los posibles valores
        max_value = float('-inf')
        min_value = float('inf')

        # Iterar sobre el diccionario
        #for key, value in Pd_pj_values.items():
            # Verificar si la lista no está vacía antes de calcular el máximo y mínimo
        #    if value:
                # Encontrar el valor máximo y mínimo dentro de cada lista
        #        max_list_value = max(value)
        #        min_list_value = min(value)

                # Actualizar los valores máximo y mínimo globales
        #        max_value = max(max_value, max_list_value)
        #        min_value = min(min_value, min_list_value)

        # Imprimir los resultados solo si hay elementos en el diccionario
        #if max_value != float('-inf') and min_value != float('inf'):
        #    print("Valor máximo en el diccionario:", max_value)
        #    print("Valor mínimo en el diccionario:", min_value)
        #else:
        #    print("El diccionario está vacío o todas las listas están vacías.")


        # Crear un diccionario vacío
        #Pd_pj_values = {
        #    1: [1],
        #    2: [2],
        #    3: [3],
        #    4: [4],
        #    5: [5],
        #    6: [6]
        #}

        #Longitudes de cada key

        len1 = len(Pd_pj_values["1_12"])
        len2 = len(Pd_pj_values["1_12"])+len(Pd_pj_values["2_12"])
        len3 = len(Pd_pj_values["2_12"])+len(Pd_pj_values["3_12"])
        len4 = len(Pd_pj_values["3_12"])+len(Pd_pj_values["4_12"])
        len5 = len(Pd_pj_values["4_12"])+len(Pd_pj_values["5_12"])
        len6 = len(Pd_pj_values["5_12"])+len(Pd_pj_values["6_12"])

        len1_1 = len(Pd_pj_values["1_1"])
        len1_2 = len1_1+len(Pd_pj_values["1_2"])
        len1_3 = len1_2+len(Pd_pj_values["1_3"])
        len1_4 = len1_3+len(Pd_pj_values["1_4"])
        len1_5 = len1_4+len(Pd_pj_values["1_5"])
        len1_6 = len1_5+len(Pd_pj_values["1_6"])
        len1_7 = len1_6+len(Pd_pj_values["1_7"])
        len1_8 = len1_7+len(Pd_pj_values["1_8"])
        len1_9 = len1_8+len(Pd_pj_values["1_9"])
        len1_10 = len1_9+len(Pd_pj_values["1_10"])
        len1_11 = len1_10+len(Pd_pj_values["1_11"])
        len1_12 = len1_11+len(Pd_pj_values["1_12"])

        len2_1 = len1_12+len(Pd_pj_values["2_1"])
        len2_2 = len2_1+len(Pd_pj_values["2_2"])
        len2_3 = len2_2+len(Pd_pj_values["2_3"])
        len2_4 = len2_3+len(Pd_pj_values["2_4"])
        len2_5 = len2_4+len(Pd_pj_values["2_5"])
        len2_6 = len2_5+len(Pd_pj_values["2_6"])
        len2_7 = len2_6+len(Pd_pj_values["2_7"])
        len2_8 = len2_7+len(Pd_pj_values["2_8"])
        len2_9 = len2_8+len(Pd_pj_values["2_9"])
        len2_10 = len2_9+len(Pd_pj_values["2_10"])
        len2_11 = len2_10+len(Pd_pj_values["2_11"])
        len2_12 = len2_11+len(Pd_pj_values["2_12"])

        len3_1 = len2_12+len(Pd_pj_values["3_1"])
        len3_2 = len3_1+len(Pd_pj_values["3_2"])
        len3_3 = len3_2+len(Pd_pj_values["3_3"])
        len3_4 = len3_3+len(Pd_pj_values["3_4"])
        len3_5 = len3_4+len(Pd_pj_values["3_5"])
        len3_6 = len3_5+len(Pd_pj_values["3_6"])
        len3_7 = len3_6+len(Pd_pj_values["3_7"])
        len3_8 = len3_7+len(Pd_pj_values["3_8"])
        len3_9 = len3_8+len(Pd_pj_values["3_9"])
        len3_10 = len3_9+len(Pd_pj_values["3_10"])
        len3_11 = len3_10+len(Pd_pj_values["3_11"])
        len3_12 = len3_11+len(Pd_pj_values["3_12"])

        len4_1 = len3_12+len(Pd_pj_values["4_1"])
        len4_2 = len4_1+len(Pd_pj_values["4_2"])
        len4_3 = len4_2+len(Pd_pj_values["4_3"])
        len4_4 = len4_3+len(Pd_pj_values["4_4"])
        len4_5 = len4_4+len(Pd_pj_values["4_5"])
        len4_6 = len4_5+len(Pd_pj_values["4_6"])
        len4_7 = len4_6+len(Pd_pj_values["4_7"])
        len4_8 = len4_7+len(Pd_pj_values["4_8"])
        len4_9 = len4_8+len(Pd_pj_values["4_9"])
        len4_10 = len4_9+len(Pd_pj_values["4_10"])
        len4_11 = len4_10+len(Pd_pj_values["4_11"])
        len4_12 = len4_11+len(Pd_pj_values["4_12"])

        len5_1 = len4_12+len(Pd_pj_values["5_1"])
        len5_2 = len5_1+len(Pd_pj_values["5_2"])
        len5_3 = len5_2+len(Pd_pj_values["5_3"])
        len5_4 = len5_3+len(Pd_pj_values["5_4"])
        len5_5 = len5_4+len(Pd_pj_values["5_5"])
        len5_6 = len5_5+len(Pd_pj_values["5_6"])
        len5_7 = len5_6+len(Pd_pj_values["5_7"])
        len5_8 = len5_7+len(Pd_pj_values["5_8"])
        len5_9 = len5_8+len(Pd_pj_values["5_9"])
        len5_10 = len5_9+len(Pd_pj_values["5_10"])
        len5_11 = len5_10+len(Pd_pj_values["5_11"])
        len5_12 = len5_11+len(Pd_pj_values["5_12"])

        len6_1 = len5_12+len(Pd_pj_values["6_1"])
        len6_2 = len6_1+len(Pd_pj_values["6_2"])
        len6_3 = len6_2+len(Pd_pj_values["6_3"])
        len6_4 = len6_3+len(Pd_pj_values["6_4"])
        len6_5 = len6_4+len(Pd_pj_values["6_5"])
        len6_6 = len6_5+len(Pd_pj_values["6_6"])
        len6_7 = len6_6+len(Pd_pj_values["6_7"])
        len6_8 = len6_7+len(Pd_pj_values["6_8"])
        len6_9 = len6_8+len(Pd_pj_values["6_9"])
        len6_10 = len6_9+len(Pd_pj_values["6_10"])
        len6_11 = len6_10+len(Pd_pj_values["6_11"])
        len6_12 = len6_11+len(Pd_pj_values["6_12"])

        print(f"1_1: {len1_1}")
        print(f"1_1: {len1_2}")
        print(f"1_1: {len1_3}")
        print(f"1_1: {len1_4}")
        print(f"1_1: {len1_5}")
        print(f"1_1: {len1_6}")
        print(f"1_1: {len1_7}")
        print(f"1_1: {len1_8}")
        print(f"1_1: {len1_9}")
        print(f"1_1: {len1_10}")
        print(f"1_1: {len1_11}")
        print(f"1_1: {len1_12}")


        # Concatenar las listas asociadas a cada clave
        Pd_pj_values_total = [valor for lista in Pd_pj_values.values() for valor in lista]

        # Inicializar el nuevo vector para almacenar los valores multiplicados por 4
        Pd_pj_values_total_hora = []

        # Multiplicar cada valor por 4 y almacenarlo en el nuevo vector
        for valor in Pd_pj_values_total:
            valor_mult_4 = valor * 4
            Pd_pj_values_total_hora.append(valor_mult_4)

        print(len(Pd_pj_values_total_hora))


        # Inicializar las 6 listas
        Pd_pj1 = []
        Pd_pj2 = []
        Pd_pj3 = []
        Pd_pj4 = []
        Pd_pj5 = []
        Pd_pj6 = []

        # Iterar sobre el diccionario y asignar los valores a las listas correspondientes
        for key, value in Pd_pj_values.items():
            if key == 1:
                Pd_pj1.extend(value*4)
            elif key == 2:
                Pd_pj2.extend(value*4)
            elif key == 3:
                Pd_pj3.extend(value*4)
            elif key == 4:
                Pd_pj4.extend(value*4)
            elif key == 5:
                Pd_pj5.extend(value*4)
            elif key == 6:
                Pd_pj6.extend(value*4)


        Pd_pj1_np = tuple(Pd_pj1)
        Pd_pj2_np = tuple(Pd_pj2)
        Pd_pj3_np = tuple(Pd_pj3)
        Pd_pj4_np = tuple(Pd_pj4)
        Pd_pj5_np = tuple(Pd_pj5)
        Pd_pj6_np = tuple(Pd_pj6)


        # Inicializamos un vector para almacenar los máximos
        Pd_pj_max = []


        #FORMA PARA LOS EXCESOS CALCULADOS CON LA RAIZ DEL CUADRADO DE LOS EXCESOS

        #FORMA PARA LOS EXCESOS CALCULADOS CON LA RAIZ DEL CUADRADO DE LOS EXCESOS

        # Crear el modelo
        model = ConcreteModel()

        # Índices
        model.p = Set(initialize=range(len(Tp_p_values)))
        model.j = Set(initialize=range(len(Pd_pj_values_total_hora)))  # Índice para j, el número de medidas
        model.j1 = Set(initialize=range(0,len1_12))  # Índice para j, el número de medidas
        model.j2 = Set(initialize=range(len1_12,len2_12))  # Índice para j, el número de medidas
        model.j3 = Set(initialize=range(len2_12,len3_12))  # Índice para j, el número de medidas
        model.j4 = Set(initialize=range(len3_12,len4_12))  # Índice para j, el número de medidas
        model.j5 = Set(initialize=range(len4_12,len5_12))  # Índice para j, el número de medidas
        model.j6 = Set(initialize=range(len5_12,len6_12))  # Índice para j, el número de medidas

        #Para cada mes
        model.j1_1 = Set(initialize=range(0,len1_1))  # Índice para j, el número de medidas
        model.j1_2 = Set(initialize=range(len1_1,len1_2))  # Índice para j, el número de medidas
        model.j1_3 = Set(initialize=range(len1_2,len1_3))  # Índice para j, el número de medidas
        model.j1_4 = Set(initialize=range(len1_3,len1_4))  # Índice para j, el número de medidas
        model.j1_5 = Set(initialize=range(len1_4,len1_5))  # Índice para j, el número de medidas
        model.j1_6 = Set(initialize=range(len1_5,len1_6))  # Índice para j, el número de medidas
        model.j1_7 = Set(initialize=range(len1_6,len1_7))  # Índice para j, el número de medidas
        model.j1_8 = Set(initialize=range(len1_7,len1_8))  # Índice para j, el número de medidas
        model.j1_9 = Set(initialize=range(len1_8,len1_9))  # Índice para j, el número de medidas
        model.j1_10 = Set(initialize=range(len1_9,len1_10))  # Índice para j, el número de medidas
        model.j1_11 = Set(initialize=range(len1_10,len1_11))  # Índice para j, el número de medidas
        model.j1_12 = Set(initialize=range(len1_11,len1_12))  # Índice para j, el número de medidas

        model.j2_1 = Set(initialize=range(len1_12,len2_1))  # Índice para j, el número de medidas
        model.j2_2 = Set(initialize=range(len2_1,len2_2))  # Índice para j, el número de medidas
        model.j2_3 = Set(initialize=range(len2_2,len2_3))  # Índice para j, el número de medidas
        model.j2_4 = Set(initialize=range(len2_3,len2_4))  # Índice para j, el número de medidas
        model.j2_5 = Set(initialize=range(len2_4,len2_5))  # Índice para j, el número de medidas
        model.j2_6 = Set(initialize=range(len2_5,len2_6))  # Índice para j, el número de medidas
        model.j2_7 = Set(initialize=range(len2_6,len2_7))  # Índice para j, el número de medidas
        model.j2_8 = Set(initialize=range(len2_7,len2_8))  # Índice para j, el número de medidas
        model.j2_9 = Set(initialize=range(len2_8,len2_9))  # Índice para j, el número de medidas
        model.j2_10 = Set(initialize=range(len2_9,len2_10))  # Índice para j, el número de medidas
        model.j2_11 = Set(initialize=range(len2_10,len2_11))  # Índice para j, el número de medidas
        model.j2_12 = Set(initialize=range(len2_11,len2_12))  # Índice para j, el número de medidas

        model.j3_1 = Set(initialize=range(len2_12,len3_1))  # Índice para j, el número de medidas
        model.j3_2 = Set(initialize=range(len3_1,len3_2))  # Índice para j, el número de medidas
        model.j3_3 = Set(initialize=range(len3_2,len3_3))  # Índice para j, el número de medidas
        model.j3_4 = Set(initialize=range(len3_3,len3_4))  # Índice para j, el número de medidas
        model.j3_5 = Set(initialize=range(len3_4,len3_5))  # Índice para j, el número de medidas
        model.j3_6 = Set(initialize=range(len3_5,len3_6))  # Índice para j, el número de medidas
        model.j3_7 = Set(initialize=range(len3_6,len3_7))  # Índice para j, el número de medidas
        model.j3_8 = Set(initialize=range(len3_7,len3_8))  # Índice para j, el número de medidas
        model.j3_9 = Set(initialize=range(len3_8,len3_9))  # Índice para j, el número de medidas
        model.j3_10 = Set(initialize=range(len3_9,len3_10))  # Índice para j, el número de medidas
        model.j3_11 = Set(initialize=range(len3_10,len3_11))  # Índice para j, el número de medidas
        model.j3_12 = Set(initialize=range(len3_11,len3_12))  # Índice para j, el número de medidas

        model.j4_1 = Set(initialize=range(len3_12,len4_1))  # Índice para j, el número de medidas
        model.j4_2 = Set(initialize=range(len4_1,len4_2))  # Índice para j, el número de medidas
        model.j4_3 = Set(initialize=range(len4_2,len4_3))  # Índice para j, el número de medidas
        model.j4_4 = Set(initialize=range(len4_3,len4_4))  # Índice para j, el número de medidas
        model.j4_5 = Set(initialize=range(len4_4,len4_5))  # Índice para j, el número de medidas
        model.j4_6 = Set(initialize=range(len4_5,len4_6))  # Índice para j, el número de medidas
        model.j4_7 = Set(initialize=range(len4_6,len4_7))  # Índice para j, el número de medidas
        model.j4_8 = Set(initialize=range(len4_7,len4_8))  # Índice para j, el número de medidas
        model.j4_9 = Set(initialize=range(len4_8,len4_9))  # Índice para j, el número de medidas
        model.j4_10 = Set(initialize=range(len4_9,len4_10))  # Índice para j, el número de medidas
        model.j4_11 = Set(initialize=range(len4_10,len4_11))  # Índice para j, el número de medidas
        model.j4_12 = Set(initialize=range(len4_11,len4_12))  # Índice para j, el número de medidas

        model.j5_1 = Set(initialize=range(len4_12,len5_1))  # Índice para j, el número de medidas
        model.j5_2 = Set(initialize=range(len5_1,len5_2))  # Índice para j, el número de medidas
        model.j5_3 = Set(initialize=range(len5_2,len5_3))  # Índice para j, el número de medidas
        model.j5_4 = Set(initialize=range(len5_3,len5_4))  # Índice para j, el número de medidas
        model.j5_5 = Set(initialize=range(len5_4,len5_5))  # Índice para j, el número de medidas
        model.j5_6 = Set(initialize=range(len5_5,len5_6))  # Índice para j, el número de medidas
        model.j5_7 = Set(initialize=range(len5_6,len5_7))  # Índice para j, el número de medidas
        model.j5_8 = Set(initialize=range(len5_7,len5_8))  # Índice para j, el número de medidas
        model.j5_9 = Set(initialize=range(len5_8,len5_9))  # Índice para j, el número de medidas
        model.j5_10 = Set(initialize=range(len5_9,len5_10))  # Índice para j, el número de medidas
        model.j5_11 = Set(initialize=range(len5_10,len5_11))  # Índice para j, el número de medidas
        model.j5_12 = Set(initialize=range(len5_11,len5_12))  # Índice para j, el número de medidas

        model.j6_1 = Set(initialize=range(len5_12,len6_1))  # Índice para j, el número de medidas
        model.j6_2 = Set(initialize=range(len6_1,len6_2))  # Índice para j, el número de medidas
        model.j6_3 = Set(initialize=range(len6_2,len6_3))  # Índice para j, el número de medidas
        model.j6_4 = Set(initialize=range(len6_3,len6_4))  # Índice para j, el número de medidas
        model.j6_5 = Set(initialize=range(len6_4,len6_5))  # Índice para j, el número de medidas
        model.j6_6 = Set(initialize=range(len6_5,len6_6))  # Índice para j, el número de medidas
        model.j6_7 = Set(initialize=range(len6_6,len6_7))  # Índice para j, el número de medidas
        model.j6_8 = Set(initialize=range(len6_7,len6_8))  # Índice para j, el número de medidas
        model.j6_9 = Set(initialize=range(len6_8,len6_9))  # Índice para j, el número de medidas
        model.j6_10 = Set(initialize=range(len6_9,len6_10))  # Índice para j, el número de medidas
        model.j6_11 = Set(initialize=range(len6_10,len6_11))  # Índice para j, el número de medidas
        model.j6_12 = Set(initialize=range(len6_11,len6_12))  # Índice para j, el número de medidas


        #print(len(Pd_pj_values[1]))
        #print(len(Pd_pj_values[2]))
        #print(len(Pd_pj_values[3]))
        #print(len(Pd_pj_values[4]))
        #print(len(Pd_pj_values[5]))
        #print(len(Pd_pj_values[6]))

        # Introducir una variable adicional para representar la expresión max(0, x)
        model.max_term = Var(model.j, within=NonNegativeReals)

        # Variables de decisión
        model.Pc_p = Var(model.p, within=NonNegativeReals)
        Pc_p_optimo = [0,0,0,0,0,0]

        epsilon = 1e-10  # Ajusta este valor según sea necesario

        # Función objetivo
        def objetivo_rule(model):
            sum_terms = 0

            # Términos que no involucran la raíz cuadrada
            sum_terms += sum(Tp_p_values[p] * model.Pc_p[p] for p in model.p)

            # Términos con raíz cuadrada periodo 1
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_1) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_2) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_3) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_4) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_5) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_6) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_7) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_8) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_9) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_10) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_11) + epsilon)
            sum_terms += k_p_values[0] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j1_12) + epsilon)

            # Términos con raíz cuadrada periodo 2
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_1) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_2) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_3) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_4) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_5) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_6) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_7) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_8) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_9) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_10) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_11) + epsilon)
            sum_terms += k_p_values[1] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j2_12) + epsilon)

            # Términos con raíz cuadrada periodo 3
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_1) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_2) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_3) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_4) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_5) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_6) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_7) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_8) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_9) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_10) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_11) + epsilon)
            sum_terms += k_p_values[2] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j3_12) + epsilon)

            # Términos con raíz cuadrada periodo 4
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_1) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_2) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_3) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_4) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_5) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_6) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_7) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_8) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_9) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_10) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_11) + epsilon)
            sum_terms += k_p_values[3] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j4_12) + epsilon)

            # Términos con raíz cuadrada periodo 5
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_1) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_2) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_3) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_4) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_5) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_6) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_7) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_8) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_9) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_10) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_11) + epsilon)
            sum_terms += k_p_values[4] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j5_12) + epsilon)

            # Términos con raíz cuadrada periodo 6
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_1) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_2) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_3) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_4) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_5) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_6) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_7) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_8) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_9) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_10) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_11) + epsilon)
            sum_terms += k_p_values[5] * tep_value * sqrt(sum((model.max_term[j]**2) for j in model.j6_12) + epsilon)


            return sum_terms

        model.objetivo = Objective(rule=objetivo_rule, sense=minimize)


        # Restricciones
        def restriccion_1_rule(model, p):
            # Pc_p <= Pc_p+1 para p en el rango de 1 a len(Tp_p_values)-1
            if p < len(Tp_p_values) - 1:
                return model.Pc_p[p] <= model.Pc_p[p + 1]
            else:
                return Constraint.Skip

        def restriccion_2_rule(model, p):
            # Pc_p >= 0 para todos los p
            return model.Pc_p[p] >= 0

        def restriccion_3_rule1(model, j):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term[j] >= 0

        def restriccion_4_rule1(model, j1):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term[j1] >= Pd_pj_values_total_hora[j1] - model.Pc_p[0]

        def restriccion_4_rule2(model, j2):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term[j2] >= Pd_pj_values_total_hora[j2] - model.Pc_p[1]

        def restriccion_4_rule3(model, j3):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term[j3] >= Pd_pj_values_total_hora[j3] - model.Pc_p[2]

        def restriccion_4_rule4(model, j4):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term[j4] >= Pd_pj_values_total_hora[j4] - model.Pc_p[3]

        def restriccion_4_rule5(model, j5):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term[j5] >= Pd_pj_values_total_hora[j5] - model.Pc_p[4]

        def restriccion_4_rule6(model, j6):
            # Pd_pj - Pc_p >= 0 para todos los p y j
            return model.max_term[j6] >= Pd_pj_values_total_hora[j6] - model.Pc_p[5]


        # Agregar restricciones al modelo
        model.restriccion_1 = Constraint(model.p, rule=restriccion_1_rule)
        model.restriccion_2 = Constraint(model.p, rule=restriccion_2_rule)
        model.restriccion_31 = Constraint(model.j, rule=restriccion_3_rule1)
        model.restriccion_41 = Constraint(model.j1, rule=restriccion_4_rule1)
        model.restriccion_42 = Constraint(model.j2, rule=restriccion_4_rule2)
        model.restriccion_43 = Constraint(model.j3, rule=restriccion_4_rule3)
        model.restriccion_44 = Constraint(model.j4, rule=restriccion_4_rule4)
        model.restriccion_45 = Constraint(model.j5, rule=restriccion_4_rule5)
        model.restriccion_46 = Constraint(model.j6, rule=restriccion_4_rule6)

        # Resolver el modelo y mostrar el log
        #solver = SolverFactory('bonmin')
        #solver = SolverFactory('ipopt', executable='/usr/local/bin/ipopt')
        


        #results = solver.solve(model, tee=True)


        # Imprimir los resultados
        #print(results)

        # Imprimir los valores óptimos de Pc_p
        #print("Valores óptimos de Pc_p:")
        #for p in model.p:
        #    print(f"Pc_p[{p}] = {model.Pc_p[p].value} --> {round(model.Pc_p[p].value, 1)}")

        # Imprimir el valor óptimo de la función objetivo
        #print("Valor óptimo de la función objetivo:", model.objetivo())
        #print("Valor óptimo redondeado de la función objetivo:", round(model.objetivo(), 1))

        ##########################################################################
        # Obtener la ruta al directorio actual (donde se encuentra models.py)
        current_directory = os.path.dirname(os.path.abspath(__file__))

        
        # Construir la ruta al ejecutable de GLPK
        glpk_executable = os.path.join(current_directory, '..\env\Lib\site-packages\winglpk-4.65\glpk-4.65\w64\glpsol.exe')
        
        # Construir la ruta al ejecutable de ipopt
        ipopt_executable = os.path.join(current_directory, '..\env\Lib\site-packages\Ipopt-3.14.13-win64-msvs2019-md\Ipopt-3.14.13-win64-msvs2019-md\bin\ipopt.exe')
        

        #print(glpk_executable)

        # Utilizar la ruta construida con SolverFactory
        #solver = SolverFactory('glpk', executable=glpk_executable)
        #solver = SolverFactory('ipopt', executable=ipopt_executable)

        #solver.solve(model)

        #################################################################

        # Resolver el modelo y mostrar el log
        #solver = SolverFactory('ipopt')
        solver = SolverFactory('ipopt')

        solver.options['print_level'] = 5  # Nivel de impresión (0: mínimo, 5: máximo)
        #solver.options['mumps_mem_percent'] = 0.5
        #solver.options['linear_solver'] = 'ma57'
        #solver.options['max_iter'] = 20  # Número máximo de iteraciones
        #solver.options['nlp_scaling_method'] = 'gradient-based'
        #solver.options['linear_solver'] = 'mumps'

        solver.options['tol'] = 1e-1  # Ajustar la tolerancia de convergencia de Ipopt
        #solver.options['output_file'] = "/content/modelo.txt"
        #solver.options['mumps_pivtol'] = 1e-6  # Ajustar la tolerancia del pivote de MUMPS
        #solver.options['acceptable_tol'] = 1e-1
        #solver.options['max_cpu_time'] = 30

        #solver.options['acceptable_obj_change_tol'] = 1e20


        #solver.options['mumps_pivtol'] = 1e-1
        #solver.options['mumps_scaling'] = 0
        #solver.options['mumps_permuting_scaling'] = 0
        #solver.options['mumps_fact_ratio'] = 100.0


        #solver.options['mumps_mem_percent'] = 100  # Por ejemplo, intenta con un 10% de la memoria
        solver.options['linear_solver'] = 'mumps'  # O dejar que Ipopt elija automáticamente
        #solver.options['mumps_icntl_6'] = 100  # Tamaño del bloque
        solver.options['hessian_approximation'] = 'limited-memory'

        results = solver.solve(model,tee=True)

        # Imprimir los valores óptimos de Pc_p
        print("Valores óptimos de Pc_p:")
        for p in model.p:
            Pc_p_optimo[p] = model.Pc_p[p].value
            print(f"Pc_p[{p}] = {model.Pc_p[p].value} --> {round(model.Pc_p[p].value, 1)}")

        # Imprimir el valor óptimo de la función objetivo
        print("Valor óptimo de la función objetivo:", model.objetivo())
        print("Valor óptimo redondeado de la función objetivo:", round(model.objetivo(), 1))

        objetivo=model.objetivo()

        #P1 manda

        print("Restricciones activas:")
        for c in model.component_data_objects(Constraint, active=True):
            print(f"{c.name}: {c()}")

        #################################################################

        # Guardar el resultado en el objeto del modelo Django
        self.objetivo = objetivo
        self.pc1 = model.Pc_p[0].value
        self.pc2 = model.Pc_p[1].value
        self.pc3 = model.Pc_p[2].value
        self.pc4 = model.Pc_p[3].value
        self.pc5 = model.Pc_p[4].value
        self.pc6 = model.Pc_p[5].value
        self.save()
