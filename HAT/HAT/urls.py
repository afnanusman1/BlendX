"""
URL configuration for HAT project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from HATApp import views
from django.conf import settings
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.index),
    path("index/", views.index),
    path("contact/", views.contact),
    path("register/", views.register),
    path("login/", views.signin),
    path("udp/", views.udp),
    path("adminHome/", views.adminHome),
    path("viewusers/", views.viewUsers),
    path("approveUser/", views.approveUser),
    path("rejectUser/", views.rejectUser),
    path("deleteUser/", views.deleteUser),
    path("approveRequest/", views.approveRequest),
    path("userHome/", views.userHome),
    path("profile/", views.profile),
    path("requestUpdation/", views.requestUpdation),
    path("updateProfile/", views.updateProfile),
    path("compose/", views.compose),
    path("inbox/", views.inbox),
    path("readMail/", views.readMail),
    path("download/", views.download),
    path("addFeedback/", views.addFeedback),
    path("viewFeedback/", views.viewFeedback),
    path('download/', views.download, name='download'),
     path('transcribe/', views.transcribe_audio, name='transcribe')
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)