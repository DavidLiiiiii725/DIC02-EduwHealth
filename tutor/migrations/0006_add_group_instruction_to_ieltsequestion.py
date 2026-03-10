from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tutor', '0005_add_reading_tips_to_section'),
    ]

    operations = [
        migrations.AddField(
            model_name='ieltsquestion',
            name='group_instruction',
            field=models.TextField(blank=True, default=''),
        ),
    ]
