import logging

import speech_recognition as sr
from nltk.translate import bleu_score

from preprocess import *



def get_stt_result(r, fpath):
    korean_audio = sr.AudioFile(fpath)

    with korean_audio as source:
        audio = r.record(source)
    result = r.recognize_google(audio_data=audio, language='ko-KR')
    return result


def main():

    DATA_PATH = './data/'
    WAV_PATH = os.path.join(DATA_PATH, 'wav')
    SCRIPT_PATH = os.path.join(DATA_PATH, 'scripts.xlsx')
    LOG_PATH = './logs/'
    THRESHOLD = 0.5

    logging.basicConfig(
        filename = os.path.join(LOG_PATH, 'match.log'),
        filemode = 'w',
        level = logging.INFO,
        format = '%(asctime)s\t%(levelname)s\t%(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()

    folders = sorted([folder for folder in os.listdir(WAV_PATH) if '어학' not in folder])
    sheet_names = list(set([folder.split('_')[1] for folder in folders]))

    logger.info(f'folders for check: {folders}')

    script_dict = {}
    for sheet_name in sheet_names:
        logger.info(f'loading {sheet_name} ...')
        script_dict[sheet_name] = load_script(SCRIPT_PATH, sheet_name)
    logger.info('\n\n')

    r = sr.Recognizer()

    for folder in folders:

        logger.info('\n'+'='*100+'\n'+'\t'+f'Start {folder}\n'+'='*100+'\n\n')
        print(f'\nStart {folder}')

        scripts = script_dict[folder[3:]]
        file_name_list = sorted([
            int(file_name[:-4])
            for file_name in os.listdir(os.path.join(WAV_PATH, folder))
        ])

        for file_name in file_name_list:
            if file_name % 100 == 0:
                print(folder, file_name)
            text = scripts.query(f'Index == {file_name}')['대사'].tolist()[0]
            text = normalize_text(text)
            text = re.sub('[ .,?!\'\"-~…]', '', text)
            
            speech_path = os.path.join(WAV_PATH, folder, f'{file_name}.wav')
            try:
                stt_result = get_stt_result(r, speech_path)
            except:
                stt_result = ''
                logger.info(f'!!!!!!! STT FAILED !!!!!!!')
            stt_result = normalize_text(stt_result)
            stt_result = re.sub('[ .,?!\'\"-~…]', '', stt_result)
            
            text_tokens = list(hangul_to_jamo(text))
            stt_tokens = list(hangul_to_jamo(stt_result))
            
            score = bleu_score.sentence_bleu([text_tokens], stt_tokens, weights=(0.5, 0.5))
            if score < THRESHOLD:
                logger.info(f'speech_path: \'{speech_path}\'')
                logger.info(f'bleu score: {score:.5f}')
                logger.info(f'origin_text:\t{text}')
                logger.info(f'stt_result:\t\t{stt_result}\n')

        print(f'End {folder}\n')



if __name__ == '__main__':
    main()
