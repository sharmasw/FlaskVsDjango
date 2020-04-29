
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
# Create your views here.
from sklearn.externals import joblib
import pandas as pd
import json

loadedModel=joblib.load('randomforestModel.pkl')


@require_http_methods(['POST'])
def scoreJson(request):
    if request.method == 'POST':
        dataToscore=json.loads(request.body)
        dataToscore=pd.DataFrame(data=[dataToscore])
        score=loadedModel.predict(dataToscore)[0]
    return JsonResponse({'result':int(score)})