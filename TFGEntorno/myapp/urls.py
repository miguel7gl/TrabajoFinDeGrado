from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('', views.index),
    path('index/', views.index),
    path('about/', views.about),
    path('calculadora/', views.calculadora),
    path('pruebas/', views.pruebas),
    path('contact/', views.contact),
    path('login/', views.login),
    path('upload/', views.upload, name='upload'),
    path('modelo/', views.modelo, name='modelo'),
    path('modelo-complejo/', views.modeloComplejo, name='modeloComplejo'),

    #Modificaciones
    path('guardar_modificacion/', views.guardar_modificacion, name='guardar_modificacion'),
    path('menuModificaciones/', views.menu_modificaciones, name='menu_modificaciones'),  
    path('modificaciones/', views.modificaciones, name='modificaciones'),

    #Login y registro
    path('login/', views.login, name='login'),
    path('guardar_login/', views.guardar_login, name='guardar_login'),
    path('guardar_registro/', views.guardar_registro, name='guardar_registro'),

    #Perfil
    path('perfil/', views.perfil, name='perfil'),

    #Pdf
    path('pdf/', views.pdf, name='pdf'),
]
# Para la ruta de archivos estáticos durante el desarrollo
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
