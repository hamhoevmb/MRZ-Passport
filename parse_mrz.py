from collections import OrderedDict
from datetime import datetime

class MRZ(object):
    def __init__(self, mrz_lines):
        parsed_lines = self._split_lines(mrz_lines)
        self._parse(parsed_lines)
        self.aux = {}

    def _split_lines(self, mrz_ocr_string): 
        return [ln for ln in mrz_ocr_string.replace(' ', '').split('\n') if (len(ln) >= 20 or '<<' in ln)]

    def _parse(self, mrz_lines):
        self.valid = self._parse_td3(*mrz_lines)

    def to_dict(self):
        result = OrderedDict()
        result['valid_score'] = self.valid_score
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
        result['check_number'] = self.check_number
        result['check_date_of_birth'] = self.check_date_of_birth
        result['check_expiration_date'] = self.check_expiration_date
        result['check_composite'] = self.check_composite
        result['check_personal_number'] = self.check_personal_number
        result['valid_number'] = self.valid_check_digits[0]
        result['valid_date_of_birth'] = self.valid_check_digits[1]
        result['valid_expiration_date'] = self.valid_check_digits[2]
        result['valid_composite'] = self.valid_check_digits[3]
        result['valid_personal_number'] = self.valid_check_digits[4]
        return result

    def _parse_td3(self, a, b):
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
        self.names = MRZOCRTranslater.apply(self.names.replace('<', ' ').strip())
        self.surname = MRZOCRTranslater.apply(self.surname.replace('<', ' ').strip())
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
        self.valid_check_digits = [MRZCheckDigit.compute(self.number) == self.check_number,
                                   MRZCheckDigit.compute(self.date_of_birth) == self.check_date_of_birth and MRZ._check_date(self.date_of_birth),
                                   ((self.check_expiration_date == '<' or self.check_expiration_date == '0') and self.expiration_date == '<<<<<<') or
                                   MRZCheckDigit.compute(self.expiration_date) == self.check_expiration_date and MRZ._check_date(self.expiration_date),
                                   MRZCheckDigit.compute(b[0:10] + b[13:20] + b[21:43]) == self.check_composite,
                                   ((self.check_personal_number == '<' or self.check_personal_number == '0') and self.personal_number == '<<<<<<<<<<<<<<')
                                   or MRZCheckDigit.compute(self.personal_number) == self.check_personal_number]
        self.valid_line_lengths = [len_a == 44, len_b == 44]
        self.valid_misc = [a[0] in 'P']
        self.valid_score = 10*sum(self.valid_check_digits) + sum(self.valid_line_lengths) + sum(self.valid_misc) +1
        self.valid_score = 100*self.valid_score//(50+2+1+1)
        self.valid_number, self.valid_date_of_birth, self.valid_expiration_date, self.valid_personal_number, self.valid_composite = self.valid_check_digits
        return self.valid_score == 100

    @staticmethod
    def _check_date(ymd):
        try:
            datetime.strptime(ymd, '%y%m%d')
            return True
        except ValueError:
            return False

class MRZCheckDigit(object):
    def __init__(self):
        self.CHECK_CODES = dict()
        for i in range(10):
            self.CHECK_CODES[str(i)] = i
        for i in range(ord('A'), ord('Z')+1):
            self.CHECK_CODES[chr(i)] = i - 55
        self.CHECK_CODES['<'] = 0
        self.CHECK_WEIGHTS = [7, 3, 1]

    def __call__(self, txt):
        if txt == '':
            return ''
        res = sum([self.CHECK_CODES.get(c, -1000)*self.CHECK_WEIGHTS[i % 3] for i, c in enumerate(txt)])
        if res < 0:
            return ''
        else:
            return str(res % 10)

    @staticmethod
    def compute(txt):
        if getattr(MRZCheckDigit, '__instance__', None) is None:
            MRZCheckDigit.__instance__ = MRZCheckDigit()
        return MRZCheckDigit.__instance__(txt)

class MRZOCRTranslater(object):
    def __init__(self):
        self.eng_to_rus = {'A':'A','B':'Б', 'V':'В', 'G':'Г','D':'Д','E':'Е','2':'Ё','J':'Ж','Z':'З','I':'И','Q':'Й','K':'К','L':'Л','M':'М','N':'Н','O':'О','P':'П','R':'Р','S':'С','T':'Т','U':'У','F':'Ф','H':'Х','C':'Ц','3':'Ч','4':'Ш','W':'Щ', 'X':'Ъ', 'Y':'Ы','6':'Э','7':'Ю','8':'Я', '9':'Ь'}

    def __call__(self, mrz_ocr_string):
        line = list(mrz_ocr_string)
        for i in range(len(line)):
            line[i] = self.eng_to_rus.get(line[i], line[i])
        return ''.join(line)

    @staticmethod
    def apply(txt):
        if getattr(MRZOCRTranslater, '__instance__', None) is None:
            MRZOCRTranslater.__instance__ = MRZOCRTranslater()
        return MRZOCRTranslater.__instance__(txt)
