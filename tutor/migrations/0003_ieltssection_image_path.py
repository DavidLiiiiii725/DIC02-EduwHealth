from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tutor', '0002_reading_models'),
    ]

    operations = [
        migrations.AddField(
            model_name='ieltssection',
            name='image_path',
            field=models.CharField(blank=True, default='', max_length=500),
        ),
    ]
