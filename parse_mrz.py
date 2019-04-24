from collections import OrderedDict
from datetime import datetime

class MRZ(object):
    def __init__(self, mrz_lines):
        parsed_lines = self._split_lines(mrz_lines)
        self._parse_mrz(*parsed_lines)
        self.aux = {}

    def _split_lines(self, mrz_ocr_string): 
        return [ln for ln in mrz_ocr_string.replace(' ', '').split('\n') if (len(ln) >= 20 or '<<' in ln)]

    def to_dict(self):
        result = OrderedDict()
        result['check_number'] = self.check_number
        result['check_date_of_birth'] = self.check_date_of_birth
        result['check_expiration_date'] = self.check_expiration_date
        result['check_composite'] = self.check_composite
        result['check_personal_number'] = self.check_personal_number
        result['type'] = self.type
        result['country'] = self.country
        result['number'] = self.number
        result['date_of_birth'] = self.date_of_birth
        result['expiration_date'] = self.expiration_date
        result['nationality'] = self.nationality
        result['sex'] = self.sex
        result['names'] = self.names
        result['surname'] = self.surname
        result['personal_number'] = self.personal_number
        result['valid_number'] = self.valid_number
        result['valid_date_of_birth'] = self.valid_date_of_birth
        result['valid_expiration_date'] = self.valid_expiration_date
        result['valid_composite'] = self.valid_composite
        result['valid_personal_number'] = self.valid_personal_number
        result['valid_line_lengths'] = self.valid_line_lengths
        return result

    def _parse_mrz(self, a, b):
        len_a, len_b = len(a), len(b)
        if len(a) < 44:
            a = a + '<'*(44 - len(a))
        if len(b) < 44:
            b = b + '<'*(44 - len(b))
        self.type = a[0:2]
        self.country = a[2:5]
        surname_names = a[5:44].split('<<', 1)
        if len(surname_names) < 2:
            surname_names += ['']
        self.surname, self.names = surname_names
        self.names = MRZTranslater.apply(self.names.replace('<', ' ').strip())
        self.surname = MRZTranslater.apply(self.surname.replace('<', ' ').strip())
        self.number = b[0:9]
        self.check_number = b[9]
        self.nationality = b[10:13]
        self.date_of_birth = b[13:19]
        self.check_date_of_birth = b[19]
        self.sex = b[20]
        self.expiration_date = b[21:27]
        self.check_expiration_date = b[27]
        self.personal_number = b[28:41]
        self.check_personal_number = b[42]
        self.check_composite = b[43]
        self.valid_line_lengths = [len_a == 44, len_b == 44]
        self.valid_number = MRZCheckDigit.compute(self.number) == self.check_number
        self.valid_date_of_birth = MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth and MRZ._check_date(self.date_of_birth)
        self.valid_expiration_date = ((self.check_expiration_date == '<' or self.check_expiration_date == '0') and self.expiration_date == '<<<<<<') or MRZCheckDigit.compute(self.expiration_date) == self.check_expiration_date and MRZ._check_date(self.expiration_date)
        self.valid_personal_number = MRZCheckDigit.compute(b[0:10] + b[13:20] + b[21:43]) == self.check_composite
        self.valid_composite = ((self.check_personal_number == '<' or self.check_personal_number == '0') and self.personal_number == '<<<<<<<<<<<<<<') or MRZCheckDigit.compute(self.personal_number) == self.check_personal_number

    @staticmethod
    def _check_date(ymd):
        try:
            datetime.strptime(ymd, '%y%m%d')
            return True
        except ValueError:
            return False

class MRZCheckDigit(object):
    def __init__(self):
        self.codes_dictionary = dict()
        for i in range(10):
            self.codes_dictionary[str(i)] = i
        for i in range(ord('A'), ord('Z')+1):
            self.codes_dictionary[chr(i)] = i - 55
        self.codes_dictionary['<'] = 0
        self.weights = [7, 3, 1]

    def __call__(self, txt):
        if txt == '':
            return ''
        res = sum([self.codes_dictionary.get(c, -1000)*self.weights[i % 3] for i, c in enumerate(txt)])
        if res < 0:
            return ''
        else:
            return str(res % 10)

    @staticmethod
    def compute(txt):
        if getattr(MRZCheckDigit, '__instance__', None) is None:
            MRZCheckDigit.__instance__ = MRZCheckDigit()
        return MRZCheckDigit.__instance__(txt)

class MRZTranslater(object):
    def __init__(self):
        self.eng_to_rus = {'A':'A','B':'Б', 'V':'В', 'G':'Г','D':'Д','E':'Е','2':'Ё','J':'Ж','Z':'З','I':'И','Q':'Й','K':'К','L':'Л','M':'М','N':'Н','O':'О','P':'П','R':'Р','S':'С','T':'Т','U':'У','F':'Ф','H':'Х','C':'Ц','3':'Ч','4':'Ш','W':'Щ', 'X':'Ъ', 'Y':'Ы','6':'Э','7':'Ю','8':'Я', '9':'Ь'}

    def __call__(self, mrz_ocr_string):
        line = list(mrz_ocr_string)
        for i in range(len(line)):
            line[i] = self.eng_to_rus.get(line[i], line[i])
        return ''.join(line)

    @staticmethod
    def apply(txt):
        if getattr(MRZTranslater, '__instance__', None) is None:
            MRZTranslater.__instance__ = MRZTranslater()
        return MRZTranslater.__instance__(txt)
