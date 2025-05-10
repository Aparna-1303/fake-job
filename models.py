from django.db import models

# Create your models here.
class Userinfo(models.Model):
    username=models.CharField(max_length=100)
    email=models.EmailField(max_length=30)
    password=models.CharField(max_length=8)

    def __str__(self):
        return self.username