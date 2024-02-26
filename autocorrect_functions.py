from autocorrect import Speller
from app_settings import settings_yaml


def autocorrect_text(text, language):
    autocorrect_languages = settings_yaml['autocorrect_languages']
    spell = Speller(autocorrect_languages[language])
    return spell(text)
