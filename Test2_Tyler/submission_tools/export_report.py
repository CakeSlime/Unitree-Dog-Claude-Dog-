"""Dependency-free DOCX plus Matplotlib PDF export of the canonical Markdown.

DOCX uses OOXML paragraphs, readable link labels and embedded PNG figures. PDF uses
paginated text and figures. Markdown remains the authoritative linked report.
"""
from pathlib import Path
import re
import textwrap
import zipfile
from xml.sax.saxutils import escape
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/tyler-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'report/ENGR6010_Second_Test_Final_Report.md'
text=SOURCE.read_text()

def plain(line):
    line=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'\1',line)
    return line.replace('**','').replace('`','')

def docx():
    paragraphs=[];relationships=[];media=[]
    def para(s,style=None):
        pstyle=f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
        paragraphs.append(f'<w:p>{pstyle}<w:r><w:t xml:space="preserve">{escape(s)}</w:t></w:r></w:p>')
    for line in text.splitlines():
        image=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
        if image:
            from PIL import Image
            path=(SOURCE.parent/image[2]).resolve();name=f'image{len(media)+1}.png';rid=f'rId{len(media)+1}'
            with Image.open(path) as im:w,h=im.size
            cx=5486400;cy=int(cx*h/w)
            media.append((path,name));relationships.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/>')
            paragraphs.append(f'<w:p><w:r><w:drawing><wp:inline><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{len(media)}" name="{escape(image[1])}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
            para(image[1],'Caption');continue
        if line.startswith('```'):continue
        style='Heading1' if line.startswith('# ') else 'Heading2' if line.startswith('## ') else 'Heading3' if line.startswith('### ') else 'Code' if line.startswith('|') or line.startswith('  --') else None
        para(plain(re.sub(r'^#+\s+','',line)),style)
    ns='xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
    document=f'<?xml version="1.0" encoding="UTF-8"?><w:document {ns}><w:body>'+''.join(paragraphs)+'<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1000" w:right="1000" w:bottom="1000" w:left="1000"/></w:sectPr></w:body></w:document>'
    styles='''<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="22"/></w:rPr></w:rPrDefault></w:docDefaults>'''
    for name,size in [('Heading1',32),('Heading2',28),('Heading3',24),('Caption',18),('Code',16)]:
        styles+=f'<w:style w:type="paragraph" w:styleId="{name}"><w:name w:val="{name}"/><w:rPr><w:sz w:val="{size}"/>'+('<w:b/>' if name.startswith('Heading') else '')+'</w:rPr></w:style>'
    styles+='</w:styles>'
    relationships.append('<Relationship Id="styles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
    with zipfile.ZipFile(SOURCE.with_suffix('.docx'),'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>')
        z.writestr('_rels/.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="document" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr('word/document.xml',document);z.writestr('word/styles.xml',styles)
        z.writestr('word/_rels/document.xml.rels','<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(relationships)+'</Relationships>')
        for path,name in media:z.write(path,'word/media/'+name)

def pdf():
    with PdfPages(SOURCE.with_suffix('.pdf'),metadata={'Title':'ENGR 6010 Second Test — Tyler','Author':'Tyler'}) as output:
        fig=None;ax=None;y=0;page=0;code=False
        def newpage():
            nonlocal fig,ax,y,page
            if fig is not None:output.savefig(fig);plt.close(fig)
            fig,ax=plt.subplots(figsize=(8.5,11));fig.subplots_adjust(left=.08,right=.94,top=.95,bottom=.06);ax.axis('off');y=.98;page+=1
            fig.text(.5,.025,f'ENGR 6010 • Tyler • November 10, 2026 | {page}',ha='center',fontsize=8,color='gray')
        newpage()
        for line in text.splitlines():
            image=re.fullmatch(r'!\[([^]]*)\]\(([^)]+)\)',line)
            if image:
                im=plt.imread((SOURCE.parent/image[2]).resolve());height=min(.34,.82*(im.shape[0]/im.shape[1])*(8.5/11))
                if y-height<.04:newpage()
                inset=ax.inset_axes([0,y-height,.98,height]);inset.imshow(im);inset.axis('off');y-=height+.04
                continue
            if line.startswith('```'):code=not code;continue
            heading=line.startswith('#');table=line.startswith('|')
            if re.fullmatch(r'\|[-| :]+',line):continue
            content=plain(re.sub(r'^#+\s+','',line))
            size=14 if line.startswith('# ') else 12 if heading else 7 if table else 8 if code else 9
            width=61 if size==14 else 73 if size==12 else 132 if table else 100 if code else 98
            lines=textwrap.wrap(content,width=width,replace_whitespace=False,break_long_words=True) or ['']
            spacing=.021 if heading else .0155 if table or code else .018
            if y-len(lines)*spacing<.02:newpage()
            for part in lines:
                ax.text(0,y,part,fontsize=size,va='top',transform=ax.transAxes,fontfamily='monospace' if table or code else 'DejaVu Sans',fontweight='bold' if heading else 'normal')
                y-=spacing
            y-=.006 if heading else .003
        if fig is not None:output.savefig(fig);plt.close(fig)
    print('Exported PDF pages:',page)

docx();pdf()
print('Wrote',SOURCE.with_suffix('.docx'),SOURCE.with_suffix('.pdf'))
