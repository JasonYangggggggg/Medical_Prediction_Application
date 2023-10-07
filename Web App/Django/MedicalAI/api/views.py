from django.shortcuts import render
from django.http import HttpResponse
from .prediction import responseLLM

# Create your views here.
def main(request):
    return HttpResponse("Welcome");

def answer(request):
    req = request.GET
    res = responseLLM(req.get("message"))
    return HttpResponse(res)

