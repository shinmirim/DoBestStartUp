from django.db import models

# Create your models here.

class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()

    def __str__(self):
        return self.title
    
class ReviewD(models.Model):
    styl_cd = models.CharField(max_length=50, default='') 
    reviewdetail = models.TextField(default='')

class Style(models.Model):
    styl_cd = models.CharField(max_length=30)
    category = models.CharField(max_length=30)
    brand_nm = models.CharField(max_length=30)
    material_cd = models.CharField(max_length=100)
    season_cd = models.CharField(max_length=20)

class Image(models.Model):
    styl_cd = models.CharField(max_length=30)
    img1 = models.CharField(max_length=100)
    img2 = models.CharField(max_length=100)
    img3 = models.CharField(max_length=100)
    img4 = models.CharField(max_length=100)
    img5 = models.CharField(max_length=100)