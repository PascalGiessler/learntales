from pathlib import Path
from re import Pattern
from typing import List

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter

from app.helper.img_helper import save_base64_image
from app.services.llm import BaseChatModel


class StoryTeller:

    def __init__(self, llm: BaseChatModel, visionModel: any):
        self.llm = llm
        self.visionModel = visionModel

    @staticmethod
    def get_context(
        audience: str, from_year: int, to_year: int, language: str = "german"
    ):
        return f"""
           You are a storyteller for an audience of {audience} aged {from_year} to {to_year}. Create an engaging and
           easy-to-understand story based on the information provided within the single hash marks up to 800 words.
           Include placeholders for up to 2 images using the following syntax: [image description for prompting].
           Inside the brackets, write a prompt describing
            the image for another model to generate. You can create up to three protagonists.
            Please write the response only in {language}.
        """

    @staticmethod
    def get_content_from_plain_text_file(
        file_path: str,
        encoding: str = "cl100k_base",
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
    ) -> list[str]:
        file = Path(file_path)
        if file.exists():
            with open(file_path) as f:
                content = f.read()
                text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    encoding=encoding,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                texts = text_splitter.split_text(content)
                return texts

        else:
            raise FileNotFoundError(f"file at {file_path} does not exist.")

    @staticmethod
    def extract_placeholders_from_text(
        content: str, pattern: str = r"\[(.*?)\]"
    ) -> [str]:
        import re

        matches = re.findall(pattern, content)
        matches = [m for m in matches if m is not None and len(m) > 0]
        return matches

    def load_from_website(self, urls: List[str]):
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()
        return docs[0].page_content if len(docs) > 0 else ""

    def generate_images_from_prompt(
        self, image_prompts: [str], output_folder_path: str | None, fake: bool
    ) -> dict[str, str]:
        if fake:
            return {
                key: "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxMTEhUTExMWFRUXGBobGRcYGBcbIBgaFxgaGBoYFxoaHyggGB8lGxcXITEhJSkrLi4uGyAzODMtNygtLisBCgoKDg0OGhAQGy0lHyUtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAKgBLAMBIgACEQEDEQH/xAAbAAACAwEBAQAAAAAAAAAAAAADBAECBQAGB//EADYQAAEDAgMGBQMEAgIDAQAAAAEAAhEDIQQxQRJRYXGB8AUikaGxwdHhExQy8QZCUnIjYoIV/8QAGQEAAwEBAQAAAAAAAAAAAAAAAQIDBAAF/8QAIhEAAwEAAwEAAQUBAAAAAAAAAAECEQMSITFBBBMiUWFS/9oADAMBAAIRAxEAPwDwVFkyT73TGHp3kSNO911VvC8cI9kZuc7grIi2GbQ4xkEYSBltZ23ZCUq2rxujtBOWiIE0O06oA759c0OtiJgazn90ITuVmtcBJ9Y17lMkDQdUG8O9Ep+4e1xv87rnvetStRMjaFzGcjOCD6EGdxSeJoz5jEkcNLaIpHFxiZtkissM7nuPhIBhIJsI4wTcCBx16FGoVLZSd95TpC0xlomSFZxsLADhqb5/G5VnXLTvchGpEXBm9ptciDxtPVVRFjuHcIAvN/xEK9OuJkgOsc+Iic8wb9Em0zAGqltSCcu80wNwK93x8ILJJN/X7qHaKzQMsp1Iy4odQ9wtRgifXLX5/tLGnJP1t6IrnQBJ6eufoD1Clpi4jI58r/VHqc69F6mGAkAi2snLdEdb7lApwJt6ze94073rnVhBjU3/AAokxOWem/Q780MH7C9N/msR3J11hNYWoSZI9knh2ebrlHwtM4WBYiNCRcj6JGg6CqVZJmYtAvw9tndwRmCQL35d9hW/Snn94AzPsjUaB4a2S9TnQRxGw03nMggRbde9oWC+oZ2oz00hanilcsa4kmDN95vkVkYKoS2ffmUMO3w0mYnZbeI+iucWWt/WMm9xrvMk8JzRMH4btPh1wcraLdHhoaIgEEXtbLcc0+eEm/Tw/ilQ4ivNPzM2QBY5nQkfPBaeAwQpvY45xeTMEZt4blvUcA1jS1gaBr5R6cNVhYyhUFQNLZYTnM2zsouSrvzw9ZhcfTd5YG1lccc0bEVocIi3svIMw5YSWOJIyET1zWlhMY4kk7rTzn46pOoNNPxdkAmNRkvKYzEHa8skAzf3y0XsKrw9sExkCbxw7CxcRgLbZEjdwSU8HlGU1xbsuMzPzoheI4I7YqMEgRkNDe4TmJEsFpOS0/8AH6QczzTMmRboVC6z0rE6X8KnYAIP/qDoBomHOhOh3lNtI9/usmrjNlxH1KyN6z0InEeI/UiADJIm2nPdZOisNwPC+/nwWXSHRO4ca2ncvTR57LMaSIcJ3xbXQ9E7h91jz462PLgg/q3mAfVGpOi6dCtjG2QMoHqisqB9oS5rdOHJXa3d7aRuhMhGN1KcAX+N32ySdZut9Z6hN0nGMuimtTJZbPl9eYKdIDoyMTYRPfD1XYWqQ2LX1tNvcZoXizHAwI3EjI8t4TeCoxfOBIz0vlHNMkT7F3GwMR0tIj7hCpgEic0Vxk8/afhVdT1BhOhXRdrzl30RSy2np05R9kKmAdmYG8wd+fE8tysMTs6HiqImy7BvzVXPOmnDP1Ufq7V55IrQI55FPgmi7DpnJGmt7BFxFIlluduSl1OI8xGuvqPhWElsWmc+AEe+uttF1IMv0zqYtF1LagttTsjOMyNY6St7wrwGpVBNgALuMADmT0CF4r4FUYG7TYa4+W4I6EZiVLVuFl80x2lu1LZ2CSRMTBykxmna9UQOQ9hf4S7mbBAcIIJBBGWi6rUmNE3XRXeHU6pnPdb7J5mKDWZXznhujnrwWUQZPx7ojMTYylcgV+md4281Igw5uWe9MeDUYABzzPeuqz8W47U3Oh668V6HwvDktA/21+6lK/kWt/ww3sDAaDkOhjlwWqyqNoCZ13LLwdN0acrIgqSJDbjS6dogmM1ngTBO/wBJtHNJ1n7QFhbd3dRUAzm/oUAEgxx9ikaD2F6bJmbd6IdCkZ4b+C1MRhhsONpiQZ77CDgSNm8d7/hZ6ZZBnxAgza35TTWbekCL3+qVw9PzTHljPd6ei0WgNaCDN7jn/YWa2aInwxagDdq2WX50P4RsLiIAOzBdBHLX3Tb8MXviRBEiy7E4OG3IEXA+eWSz3Rq44IqYkAbpNstclmumbwr1SHRefVV2enBQZslHjf0jEjvhwsr0yAQSJEGxncdx0JBQKVQG0wTkSYiTNz9V1OoMtV6qPJY2yoIRWO7KBs788s8jOs9RCn9Qp0xWh6kQc+/T5TTcRaAs2jUiPjr8or6hzA+qdCs2WuEdEFuIuRCWoVH7MacPS/ojPsyQPNI+YVZIUCc3aJB4QmcRTaBIN8t2vuEOcvLu7KVxTiTF43cQOW9OTWhWUzNte5TL8Nbhv+O+CvgmgMyva/qmG1bEb80yBXwSFIfH55KK2HkDUXg9dUwWA5f0oq0iBw1IHcZJ8FVCdHCAD/bhFuRnmqtpmY0WthW5jgg1mRMDPcO9U8MS/ngKnRjSfzyT9PB2mCefyUEscyA4EG2Y38wvUeEUttsRIAz039JRprNEnfh5D/NsZVIZhwXNpBrDs5Bzi2STH8iARwG6Vn+A4r9Go7DsIrUHjaBvLHAXcJA5G3rCe/y2vtYysKboYwBhbfRrdtttAWjhI4LOwVBrHgjMgmx0iYheZKbrf9PYblcef4N4nDuIa6wBG8aEjKZGWqijQ2srwJ5RqiUpeIsL7wM7CJOQ3+q7D0Y+b2nS0r00/DymvdFqtKLyBcSJvfd6ZpfExs2yiTabpnFUje6Bj2bLbTcxoDB1hK/hyb3wya4Lz5RkJNlr+F4U2MZXsg+HFrgNrL0J00WjhqIbMGATn9PcKKn3S1cnmM28PSJh4IFriL9bJloDRfXL7KmApw0AxG8b9yvUdaM2/CLfhN7uoXxVI6Lm0fLcZ5TuhWMzG+03R8RRIAGU3U6HQhVrWcCNPlQ3Cua1pvsOu074MHLjbor/AKQ2iDr3ZVpg/wAActVlsvLH8OZaIE/myKIuMj9Et4eSLeybqU/MCdcx/SxclG3hTYJtbZME5ZclFd21MzfW6vXaPsu2bDkDHeqyt6b5nDKdh9m+iJSYIuJ9EfGVItpqOXJZj8SWmAbJfpefD5lQrum8ngmadUu5jTWBvUUWw0EQbXG4zlxyBViQDJaQYgr1jyDRw+IPCe7ymwd4M6TqLjLmDfgsjC4u/sny+THA/REVjsAfhFd5QLwTx6oAdAEnl2RdL06geTcnZiPt6JkxGtNPDYm0aora08PjuyzG1CDGiMyq2TmRobDkYv6e6smTqdNGnWzB0y7hDqnf6cEi6ts5H4+6qPEdxvM3vkZ1Xdjuhs06w2RB0umKL4IPD6QFk0CXCS4AXvB6Cw1yy9FemXQY0Tqibg14vLTHHLj1U1aZjPLPqqUA4Z2jrPDiuqE9O9yqqIuR3B4MEbUmQTFxfK0Rz11TNR7GAOY0ybbRO8EQByka+6yqWINrZZfhNGqX7O17fC5rWcn4amEIqDZc02yOumaz/Hf8lq4Oq3C0NmmNlpfVIl3mziZDQMstFoeFgOfDfKJPS1ze+QWHUAqY59Wo3bZtATAfYbNmgmJt8rPy/wDJfh+9v6M7E4c0y4ujbdukh7f+d7kk3JIkypwlKakiwJjLLd7T7r0WLwJJBqA7UDavlbIeiaw3hw2Q62Z53HwmmcQvJya2zzzsOQSIFjPQxExy+VFJvmgx6Hn+FvVcLJECBr9uOfuEvjsPPmGkblpT8MtCFWmImIPJYnibSSNlpIbJOcAZSY4n1IW+wSTw0+6Tq0I2+Iy3rqeoEtJ6ebw7ycjBGe+y1MBVmxniChfsNmAGy6JOduPJN4TDAPg5Z2vp9wFJaVppm7hqcGJscp+UzSqSM/8A5jXXkq/si1gk3zXeB1Q6oWkaH3i4QqkBJ6Vqtc4W32gFHw9Tyybxa9k6+gBnl3Cya3lLvXhKhVlFJbEtgzGeSBTcWun0+NFotpbbSXOkwh02NiNq43LNyWi/HL0YwOHJcDGvdl2PBDrZRYd8imvDXwb59joh+KQHZXO7vuV5916erxT/ABFqplsjNJUsQRnmFoUnACN3vKy8bUaCbchxUzTKEcfiDc9VnPxgGWXTsKce4zE2N0i7a0aSFSUFvDDpgboGmu7XJENIRxH17KhuiL+roO/vmfVehp5L+mbiaEXCmliU459762zWfUbnGqJxrPxkGNqAWgHjkcuYBVMC0bJdlJPtksnzPc1gz3rYDAGxutHLeimc0EDgdc/bK89fZdWcLRIy/Kow3JFo7hAeSdU+gwM6sHbh3xVsLEyEsITVOllHX1OXCIR07Dbw4GyJ75dFfDt3X4d8lmirAhQzFO0E2/E+qbSeaeko1Z1ygad6fCNVcJgEfleX/f7JmbaymsNiCW+W5OQFrzN+Hqj3wR8enoKdO8AycxBGqeoYYC+ixPD8c4DTaBuOW5aA8TZIE3dkYtO4cUf3RP2zaog0mOLBtPcQxgABnV2emh5pbwii3YYC0CCfNvcJBPr0Wj4fD3NIiGtN3NG0TJE2JAAMHPejspN2XMBZNP8Ak3KxM7Q4EO04rHXJt6zdPFnHiA4rCDaaf5Hgc4ujYgDZ8ojnE2WdiMcWxNmiwOXD6oVfxKB/H+U6/ZWjmZl5OJIXr1zNyLe/3QXV7Z8PwlamJa69jcZadx8ob9vatEHqtM8pmriSNLCUo80WieipiGNJtGeehUh7tgEzPdlX/qbxcGDMi8WteUf3NJ/t4ZniLA2pnYjsJCoNhzXAyNfVGxlY5HQ8+GqXfiI02gldhUnpqWO/VAkjKAlKL30jIvGeeSWwjQ4bbSYEb7c9yYo+fWxlRqyik3KGK272gR3CpjqYcIFyUrgG7NhedE+GAZuk9LLPXJhWOPWI0WEy3dnbMcknSr7DoAInhK1BT88gxIv3qUvi6diQIIPxuWa+TTbxcWI0/CagcbxOiL43RgtI6pTwzEiGjLX0TWOxTSMlkt+noccYjOqVQLDJYniGc8U7Xq34/lZ2NdaJ73IyVSEMQ8CJHTgrMoGLX6utwsQhPIPmJOfY73o5c2Tc9I+qstXwHjZ5R+vAq9PPcqBs9VOJMNIGuq3njkMAcfhd+0HGUXC04AvfVOviNyG+hfwUw7GtERfUjVFNPTv1QQRtADNNVCG+Z0wT6k6e0pxNZWnhxqESpREbgjtbzjd3wJQ6sgC351n0IRR2iFAS52sGE7aQQDHHvmh0KQDjpvRdnv6Jkc2xks3i9pt1+ECpNzJJN5PO59ZS76hm1ldlyEwAJeZvELTpy2HNjIDSI65LPzOc+9uwFp4d1oMSN87zYj7IMI/4eym4gk3tMmLz8zC234elEhgI3W5ZryhMOsQQTPEfZamBxpcRIlupysNx3qdSzk0jZwOKh+y1nCCY9zlzTTaGxWpkCf1XNZGzcAGZDjwsRqk6T2gkl0BsSb77TlvhX/xvxCq7Ei23TJdYAeWSJeLSbwd/qVj5fH4eh+nxr0dxVBzqpsNhoJLYOZyEncFj+IbRBAjM8/tFl6fF/wCRuoVHNqsZUY+wLXECPcExwnNB8YpMp7X6YDgRmjxcukv1HE5PK4HBmbmE8aAEX90riceBJLecfRXo40PEgkc59+C1yzzrNagQW3+UKtTaAdFIa2CSdBEZHgd1kpisQ0CZ+NeGf9hF0J1ZhY9pLoiyWw+EJdsyfuncZiA4z0RMPUBbZsHU/EIVeBmNZenRNNthY58U7haADZJS2KxLrbhlxQv3RJy97AdfqpuvB1D038PUHl1ju6tWrt2vLJ38AsSjjCBHonMKQWh215jMi9oNr6/hZrNnFBqseyxPqkziBkMoj0sl679BmEpSc8ndf1Wdm2JGaVYtdaOoB1Bz6fO9MjEQTJv8BZmOaYBHlj4WYzFwbETn2V2di6yQ/iWNzAWW/F8ZXYyrJJI5rKdU82duP1hWiDPfKPVa834390Q4s6ALNDgdOf4Cr+9iwTOTlZBImBuXMd1ul21ha+/imqQuJm8aaarSYcG6VLWO+wVx5yr0HCDB4dhDqPGQNvS+qGgc4ZeIcQ86QCD1/CewNIEtc7IDW9+yUgGEkkiAdDzt7I1QwM8txyv/AF7JtOw1KVXanasMhHD59lc03OBMWaBJkTeYsewAFnNeSJmORGoOnenBWGJJMD1PAJkLn9hSdkb1Q1CWm+UQOBm+61vVVpiUzUoiLJlQMFg700n355orKtrKH0LZ5/PLehtbu77lNoAofwR2v3x6Zxb7qlKjMkaXN+MdcwmaeHldorWE1H7QFoAEAwBMchfPM3TGDw5J2ROWQm5+pKXDIMR6c07h9B3CWq8OS1mrgGgMIcRpnoRlzStfxp9JsMgOcSC8aT5THQlFZhwSBecjPt7ysrF4fz7JIidMrHTdkslY36beNOV4Mf48wPrsdXef02vBcXSQBNrZ5/de+8YY0AhjttoOYnIiwnWwXgv02vqBsRJgxoIgEjevW+EscxmyDZ2c3BOUXyP2CjTx6WcNz6K06THgiDPLuUu7wlrSSZI/4g5zkvQOoBpySVekDMm0bvY3Whcnh59cTdennKRcXlrZaJy56BRi6YBh1vexT9YtaYj8bln4o7Ud6ru5y4hYuYIyke/NEoYgHgeQhJVaGzn3PFDrmLhdujdGh/EOBAg3UNot2eXys5mMc43kmwE7h+E5ReYcCM0KDCLU26SU010Dhv3cPdLsccozS/iPiNGnVfSIqHZsXQIByMNnT7qT1mmMQ/SO1JcTI4owxjWtJvPPJZnh2MDiYDgDGzIHm3yZtpA53U4iv/qd9z9FNzrw1RSS1A8X4gY2fXros0Vb7p4+qLiHAPcWTsgmNqCYOU6TGoSdZwi4g/fJVmEiN26GKtxcgceQlIuByzUGvoRaRMGLcM4K5pnvK8Xtc5J8xCbrKPEDOZ5pcsPBOvbyK6jW2RET0b9QUExnJnsCKH6H1lBDsvv3Cp+rfPsjL3hXMqNEVI17Kg1hJ+Ul+4HeaE95ce+zmuw4llWxGkn50TDnzFum7olgIRGNJy0+EThmnVuC0Qcuc2TVGkHEXDZtLrAc7TCz6NhH1R2uLTwtnqh8Oa02sLhd8I37LaAjQGfVZ2Cr2g52i/qmBiC0Hu6Vuh0ozGNtwuy0kX05cShMwog7yhYes5wNjBzPVOUDEEG/FDszuk/g7DYcCxAKfGFJG4AZTqB9UFjTN0alioGzvXO2Bcc/kj9rB05c022g0gyYOkNEWiOU3v8AdL/uCbWlSKpSumMoS+DuGsdEpj6Ow0umXum0fxiM5Gs6LQb4Y+A4kDddt/dK+O0SGEC+8/MblFv00TOI8/4TUJq7TiZcZ9Sveh3/ABvEZcc14DC6b5XtcNWAkzNhkdbeq7kXosP+LNB/iZYBMSbX5LExeMlxvbfKjFYoGQTnHtlfRZuNIhGUQp6WxFbbMyT/AHCUq1/RVZVgRC4sJGVhbTWSJOv4T4BN4c4Ejp3z0S0EEBabKQDBntbUER/rGc6f6oT2NMkCI0mTzNkOwzhtCraHm/jppPf9qa+LFOILKkOgtDoNxpaSZHTqtQBthyS3/wCMAdlogTLTzOXDhyQ7r8hXG/wV8P8AERUJDqf6Z/1E57wbC/RZ/jLmOeHAgznaJMwTxtuWzR8Ia8vbBkEWuIcW/wBXWf4lhWANg7Tm7rab98pFc9vDQuKnPonVwLok5D6ajcgue4NI2ja987358U/4hjDstO1bJw562zyWRWxAkhGW2Ncqfg6yoC0EcbpKtfP+1ShXAiFaSb+2ab8ifUUFEnS0omwAL5oVWs7MWjdp1VWucSO7ItNnJpF67ibngOgFvYIdSmWktLbjMbkZ4AyJ9O5V2Vdm20R/1G/nF9Oi5HUY06AzfJcGwbiCJkGcxodQhtcAczBiYOkgn4noruNhAMbzqRnHQi33WkyEQJuj0mAoMxJ3hVY9KwoYrUwpAnIKgfKL+pa39dVyZzQKoCDEIrHKC4H6LnmEQYNinEXm3G1zb69UUEk52SmHq5i5gabt6LTxPAC0JdD1TNalAG5Vo4k35cOQ90ttyqGp3CVDPz4aped4lS5yzqWIPY32TH7mRO4X+Pqu6tC9kaNJwBvyhF/chpnvuyyKT7gA/Qequ2vYOkSDlraD9fnch1D3Y+zGuc65MkhO+I4v/wAThAMA33HLraFj4d/mnWUPE1XAxpqN6RytK8dtJ6UwmMex7XtMOadoHO4uD6rVp4owCTebkLC1k3HeS06MFkDu4RpCzTNBhnTqiOGvHLlv9vdUw1QBnLPjClhJ6wk0foLYlrn6XA0GQsNOJHqiYHAFzS64vAEG5R8H4Y55BPlE3LrRrPFOeNucKYZT/hebQXTmTuFohLXJniDPDrbYHFgMdTY47RcYkQQwxIBM5nK0/Y1fCtaS5ojayE5XyJ6LDxFR5YA0gAOBda9sgN11oDxPygR6qdbheYQ9Xp0wJk3ygHpM6LmVmM8r7xcFsaLKfjnOEbVvnggHEgA3P30SNNlVKQ54l4o3SQTcnfHwsbFY4QZddAxdYE2EWWTiqqrx8YnJy4gtfEGDcELPfUJQ3P4qWEjr73+4WuY6mGrdB6RvfqMj+Cm8OQTslwaP+RmB6SUthwBn3p9Z6IrqeqWh58RbNHa2EKIjI62V556/FkjHRD6g498O8lVtTgOod9CiU6UqRRRTwDTZhQuac/a034q7/wCI9hy6yMydEL274rQZg5jMSGmYkgm0ZkZ5i8CVV1tImD7H5kHu1AVIdeeqUJem86WTFO5gcALBukCYtPHmTKXA6IgeOf3tnv1TChoi837yV4kc/pw0S7TYmdMr3uB9T6cleoQCQDtDeJg8RtAH2CGBTCUxAy5HdCmi2e+9ymmAYkwN90VoAE/dAJLKpFgTCIGyQM+WaC2oIHBcalkRG9GGADPIZib20yzzUPfOXolHv779FD6rgA05Z5AZgCeNgLp8EaGG1DkNUzhxy+3RZ7CVo4VsEEnRJXiGha8NGmYglK168yCM8j624/gIWLxhcRdK1K0kmd3fe5SlP6aaazEaOKxTHEuaz9MQ2GgkxAAJlxm5k9VWliRMEjQd3SbAIM7uHeiHQDbz0+/si8EWnoMPUJtqU6zxBzPKGi2Z16LCwtaB/KIFoU1cYYzO7fzWdzrNctStPR0cU+pcggZSTuXeIYkGGzl32F57DYoxnbgr1MSG5dVNx6VVpoaLpJBsF2KewCxJ6ax90CnXB1ieO+14uPRZ9bG7JBEWNpuJHA2KZS2znSlFn4wxHvf03aJd+LixdKWqYgGe/wCkqXSrrjRlfKxuribad697ko691fZtl1VRl7J0sEpuvoIsRmU4z76qAe+iIGxzRbYqSL02ZoobZdh2AlomAYuZgSYk6mOA3orxe5mJvpE5ibxmcgpsqsK7EQiN/PfouZECdFP6oHK/p3olY6DMEaKrzqYvzQX19rh3kmaDWuEuqMaZyftzvnysda6CQWzzMnJWaLEyIJiJg6Gdmcuduq5ctRiIBv13T7HPkuYYzAyi824i4uuXLgkMdorAxz3Lly4Uknll2b6qWP4KVyY4NtRa3yEba5FcuUxir9Jnu4UOBN1y5URNlsTs7R2J2TkHRtD/ALRaeSG5cuRAGpaA2nWPfiiiqZsoXJKHllqtX47y5peq6/BcuSoeizXmI65X9c1LDeZhcuSsZFnuVi/pOl/RcuQSGbDCoIEW+t5n6dFbEVmgNIc7a10gzbZMybcrqFyXPRtxCv7gjIGMuu5L139f6UrlVSkSdNoC0HOPaYv+IRKDJyXLl1Pw6V6Ge0QO7qrYULkiHf0kkWtzj+lxce/ouXInBGuJNr6bzuH2U/qFcuQOLMrRecoyz6HRUdWMETYxI3xvHquXLgkNqDZAETJORkWFiciLW6qr3FcuQxB7M//Z"
                for key in image_prompts
            }

        images = self.visionModel.generate_images(image_prompts)
        for key, value in images.items():
            if output_folder_path is not None:
                file_name = key.replace(" ", "_")
                save_base64_image(value, f"{output_folder_path}/{file_name}.jpg")
        return images

    def transform_text_to_html(
        self,
        content: str,
        images: dict[str, str],
        model_name: str,
        with_llm: bool = False,
    ) -> str:
        for key, value in images.items():
            content = content.replace(
                f"[{key}]",
                f"<img src='data:image/jpeg;base64,{value}' alt='{key}' "
                f"style='float: left; margin: 10px; width: 150px' />",
            )
        if with_llm:
            result = self.llm.invoke(
                model_name=model_name,
                prompt=f"Adjust the following content in single quotes to be valid html and return it as string: '{content}'",
            )
            return result.content

        return f"{content}"

    def tell(
        self,
        original_content: str,
        model_name: str,
        audience: str,
        from_year: int,
        to_year: int,
        language: str,
    ) -> str:
        text = original_content
        result = self.llm.invoke(
            model_name=model_name,
            prompt=f"{self.get_context(
            audience=audience,
            from_year=from_year,
            to_year=to_year,
            language=language
        ) + original_content}",
        )
        content = result.content
        placeholders = self.extract_placeholders_from_text(content)
        generated_images = self.generate_images_from_prompt(
            placeholders, output_folder_path=None, fake=False
        )
        html_content = self.transform_text_to_html(
            content, generated_images, model_name=model_name, with_llm=False
        )
        return html_content
