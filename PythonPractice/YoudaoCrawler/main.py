import requests
import time
import random
import hashlib

class YoudaoTranslator:
  def __init__(self):
    self.url = 'http://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule'

    self.header = {
      "Accept":"application/json, text/javascript, */*; q=0.01",
      "Accept-Encoding":"gzip, deflate",
      "Accept-Language":"zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
      "Connection":"keep-alive",
      "Content-Length":"255",
      "name":"Content-Type","value":"application/x-www-form-urlencoded; charset=UTF-8",
      "Cookie":"_ntes_nnid=30f20fd34ac064b923a7f3d56eeef45e,1562765868764; OUTFOX_SEARCH_USER_ID_NCOO=935754296.4193134; OUTFOX_SEARCH_USER_ID=-179545565@10.168.11.24; YOUDAO_MOBILE_ACCESS_TYPE=1; JSESSIONID=abczrgXeofqGe6PxVEZVw; ___rl__test__cookies=1563176774538",
      "Host":"fanyi.youdao.com",
      "Referer":"http://fanyi.youdao.com/",
      "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:68.0) Gecko/20100101 Firefox/68.0",
      "X-Requested-With":"XMLHttpRequest"
      }

    self.data = {
      'i':None,
      'from':'AUTO', # 源语言语种
      'to':'AUTO', # 目标语言语种
      'smartresult':'dict',
      'client':'fanyideskweb',
      'salt':None,
      'sign':None,
      'ts':None,
      'bv':None,
      'doctype':'json',
      'version':'2.1',
      'keyfrom':'fanyi.web',
      'action':'FY_BY_REALTlME',#'FY_BY_CLICKBUTTION'
    }
  '''
  # 对应JS代码
  var r = function (e) {
          var t = n.md5(navigator.appVersion),
          r = '' + (new Date).getTime(),
          i = r + parseInt(10 * Math.random(), 10);
          return {
            ts: r,
            bv: t,
            salt: i,
            sign: n.md5('fanyideskweb' + e + i + '97_3(jkMYg@T[KZQmqjTK')
          }
        };
  '''
  def translate(self, src):
    self.data['i'] = src
    nowTime = int(round(time.time() * 1000))
    self.data['ts'] = str(nowTime)
    self.data['bv'] = hashlib.md5(self.header['User-Agent'].encode('utf-8')).hexdigest()
    self.data['salt'] = self.data['ts'] + str(int(10 * random.random()))
    self.data['sign'] = hashlib.md5(('fanyideskweb' + src + self.data['salt'] + '97_3(jkMYg@T[KZQmqjTK').encode('utf-8')).hexdigest() # 16进制
    res = requests.post(self.url, data = self.data, headers = self.header)
    return res.json()['translateResult'][0][0]['tgt']

if __name__ == '__main__':
    translator = YoudaoTranslator()
    res = translator.translate("Hello World !")
    print(res)
    
