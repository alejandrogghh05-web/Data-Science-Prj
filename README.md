# Proyecto: Precio de la vivienda en Melbourne

## Dataset

**Melbourne_housing_FULL.csv**: 34.857 registros y 21 columnas con información de viviendas vendidas en Melbourne (Suburb, Rooms, Type, Price, Distance, Landsize, BuildingArea, YearBuilt, CouncilArea, Regionname, entre otras).

## Enunciado

**1. ¿Podemos predecir el precio de una vivienda en Melbourne con una precisión razonable utilizando sus características y los modelos vistos en clase (regresión lineal múltiple, Random Forest y CatBoost), evaluados mediante validación cruzada?**

**1.1. Predice el precio medio de una vivienda**
Comparar los tres modelos con validación cruzada (k-fold) usando RMSE, MAE y R², y valorar si las diferencias entre ellos son significativas o se deben al azar entre particiones.

**1.2. ¿Qué características encarecen una vivienda?**
Obtener los coeficientes de la regresión lineal múltiple (con variables estandarizadas), la importancia de variables de Random Forest y el *feature importance* de CatBoost, y comprobar si los tres coinciden en las variables más influyentes (superficie, número de habitaciones, distancia al centro, barrio, año de construcción, etc.).

**1.3. ¿Qué tipo de vivienda (casa, unidad o terreno) genera más beneficio?**
El dataset no incluye costes de construcción ni de compra, así que habrá que definir "beneficio" como un indicador proxy, por ejemplo el precio medio por m² o el precio predicho por m² una vez controlados superficie, barrio y antigüedad. Después, comparar el resultado entre tipos con cada modelo.

**1.4. ¿Es rentable un terreno sin construir?**
Entrenar los modelos solo con las viviendas que tienen información de superficie construida y estimar el precio de terrenos sin construir a partir de su superficie de parcela (*Landsize*), ubicación y resto de características. Comparar el precio estimado con el precio de mercado de la zona para valorar si la inversión tiene sentido. Esta pregunta requiere supuestos explícitos sobre el coste de construcción, que habrá que documentar.

## Modelos

- Regresión lineal múltiple
- Random Forest
- CatBoost
- Validación cruzada (k-fold) para todos los modelos

## Entregables

- Comparativa de métricas (RMSE, MAE, R²) de los cuatro enfoques
- Tabla de factores que encarecen la vivienda, con la coincidencia entre modelos
- Definición explícita del indicador de "beneficio" usado en 1.3
- Supuestos y conclusiones de la rentabilidad del terreno en 1.4