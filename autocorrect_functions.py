from autocorrect import Speller


def autocorrect_text(text, language):
    from init import settings_yaml
    autocorrect_languages = settings_yaml['autocorrect_languages']
    spell = Speller(autocorrect_languages[language])
    return spell(text)
