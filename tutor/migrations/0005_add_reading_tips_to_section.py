from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tutor', '0004_add_group_label_to_ieltsequestion'),
    ]

    operations = [
        migrations.AddField(
            model_name='ieltssection',
            name='reading_tips',
            field=models.JSONField(blank=True, default=list),
        ),
    ]
