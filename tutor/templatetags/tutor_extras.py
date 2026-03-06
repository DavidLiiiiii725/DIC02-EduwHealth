from django import template
register = template.Library()

@register.filter
def get_item(d, key):
    if isinstance(d, dict):
        return d.get(key, None)
    return None

@register.filter
def multiply10(val):
    try:
        return int(float(val) * 10)
    except (TypeError, ValueError):
        return 5

@register.filter
def multiply100(val):
    try:
        return int(float(val) * 100)
    except (TypeError, ValueError):
        return 50
