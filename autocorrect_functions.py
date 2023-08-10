from autocorrect import Speller
import settings

autocorrect_languages = settings.autocorrect_languages


def autocorrect_text(text, language):
    spell = Speller(autocorrect_languages[language])
    return spell(text)
