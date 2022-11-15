from django.contrib import admin
from .models import JabRefIssue, PowerToysIssue, AudacityIssue


# Register your models here.
admin.site.register(JabRefIssue)
admin.site.register(PowerToysIssue)
admin.site.register(AudacityIssue)