\NeedsTeXFormat{pLaTeX2e}
\ProvidesClass{ise-thesis}[2008/01/29 情報システム工学クラス卒業研究報告書]
% jarticleのロード、オプション処理
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{jbook}}
\ExecuteOptions{12pt}
\ProcessOptions\relax
\LoadClass{jbook}
% パッケージのロード
\RequirePackage{times}
\RequirePackage{graphicx}
\RequirePackage{citesort}

% サンセリフ＋ゴシック体 (\sffamilty\gtfamilty)
\newcommand{\gothic}{\sffamily\gtfamily}
\newcommand{\gtfont}[1]{\begingroup\sffamily\gtfamily#1\endgroup}

% itemize のラベル
\renewcommand{\labelitemi}{・}
\renewcommand{\labelitemii}{\scriptsize ○}
\renewcommand{\labelitemiii}{\scriptsize ●}
\renewcommand{\labelitemiv}{\scriptsize □}

% 箇条書きの行間を調節可能に
\newdimen\@LSLen
\newcommand{\liststretch}[1]{\@LSLen #1\p@}
\liststretch{1.0}

\setlength\partopsep{3\@LSLen \@plus 1\@LSLen \@minus 1\@LSLen}
\def\@listi{\leftmargin\leftmargini
  \parsep 4.5\@LSLen \@plus2\@LSLen \@minus\@LSLen
  \topsep 9\@LSLen   \@plus3\@LSLen \@minus5\@LSLen
  \itemsep4.5\@LSLen \@plus2\@LSLen \@minus\@LSLen}
\let\@listI\@listi
\@listi
\def\@listii{\leftmargin\leftmarginii
   \labelwidth\leftmarginii \advance\labelwidth-\labelsep
   \topsep  4.5\@LSLen \@plus2\@LSLen \@minus\@LSLen
   \parsep  2\@LSLen   \@plus\@LSLen  \@minus\@LSLen
   \itemsep\parsep}
\def\@listiii{\leftmargin\leftmarginiii
   \labelwidth\leftmarginiii \advance\labelwidth-\labelsep
   \topsep 2\@LSLen  \@plus\@LSLen\@minus\@LSLen
   \parsep\z@
   \partopsep \@LSLen \@plus\z@ \@minus\@LSLen
   \itemsep\topsep}

% 見出し前後の行間を調整可能に
\newdimen\@SSCvs
\newdimen\@SSCdp
\newdimen\@SSex
\newcommand{\sectionstretch}[1]{%
  \@SSCvs #1\Cvs%
  \@SSCdp #1\Cdp%
  \@SSex #1ex}
\sectionstretch{1.0}

\def\@makechapterhead#1{\hbox{}%
  \vskip2\@SSCvs
  {\parindent\z@
   \raggedright
   \reset@font\huge\gothic
   \leavevmode
   \ifnum \c@secnumdepth >\m@ne
     \setlength\@tempdima{\linewidth}%
    \if@mainmatter
     \setbox\z@\hbox{\@chapapp\thechapter\@chappos\hskip1zw}%
     \addtolength\@tempdima{-\wd\z@}%
     \unhbox\z@\nobreak
    \fi
     \vtop{\hsize\@tempdima#1}%
   \else
     #1\relax
   \fi}\nobreak\vskip3\@SSCvs}
\def\@makeschapterhead#1{\hbox{}%
  \vskip2\@SSCvs
  {\parindent\z@
   \raggedright
   \reset@font\huge\gothic
   \leavevmode
   \setlength\@tempdima{\linewidth}%
   \vtop{\hsize\@tempdima#1}}\vskip3\@SSCvs}
\renewcommand{\section}{\@startsection{section}{1}{\z@}%
   {1.5\@SSCvs \@plus.5\@SSCvs \@minus.2\@SSCvs}%
   {.5\@SSCvs \@plus.3\@SSCvs}%
   {\reset@font\Large\gothic}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{\z@}%
   {1.5\@SSCvs \@plus.5\@SSCvs \@minus.2\@SSCvs}%
   {.5\@SSCvs \@plus.3\@SSCvs}%
   {\reset@font\large\gothic}}
\renewcommand{\subsubsection}{\@startsection{subsubsection}{3}{\z@}%
   {1.5\@SSCvs \@plus.5\@SSCvs \@minus.2\@SSCvs}%
   {.5\@SSCvs \@plus.3\@SSCvs}%
   {\reset@font\normalsize\gothic}}
\renewcommand{\paragraph}{\@startsection{paragraph}{4}{\z@}%
   {3.25\@SSex \@plus 1\@SSex \@minus .2\@SSex}%
   {-1em}%
   {\reset@font\normalsize\gothic}}
\renewcommand{\subparagraph}{\@startsection{subparagraph}{5}{\z@}%
   {3.25\@SSex \@plus 1\@SSex \@minus .2\@SSex}%
   {-1em}%
   {\reset@font\normalsize\gothic}}

% 用紙サイズの設定コマンド
\def\paperSize#1#2{%
  \@paperWidth0pt\@paperHeight0pt%
  \edef\@paperSize{#1}%
  \def\@sizeName{a3}
  \ifx\@paperSize\@sizeName\@paperWidth296mm\@paperHeight420mm\fi%
  \def\@sizeName{a4}
  \ifx\@paperSize\@sizeName\@paperWidth210mm\@paperHeight296mm\fi%
  \def\@sizeName{a5}
  \ifx\@paperSize\@sizeName\@paperWidth148mm\@paperHeight210mm\fi%
  \def\@sizeName{a6}
  \ifx\@paperSize\@sizeName\@paperWidth105mm\@paperHeight148mm\fi%
  \def\@sizeName{b4}
  \ifx\@paperSize\@sizeName\@paperWidth257mm\@paperHeight363mm\fi%
  \def\@sizeName{b5}
  \ifx\@paperSize\@sizeName\@paperWidth181mm\@paperHeight257mm\fi%
  \def\@sizeName{letter}
  \ifx\@paperSize\@sizeName\@paperWidth8.5in\@paperHeight11in\fi%
  \def\@sizeName{a4r}
  \ifx\@paperSize\@sizeName\@paperWidth296mm\@paperHeight210mm\fi%
  \def\@sizeName{a5r}
  \ifx\@paperSize\@sizeName\@paperWidth210mm\@paperHeight148mm\fi%
  \def\@sizeName{a6r}
  \ifx\@paperSize\@sizeName\@paperWidth148mm\@paperHeight105mm\fi%
  \def\@sizeName{b4r}
  \ifx\@paperSize\@sizeName\@paperWidth363mm\@paperHeight257mm\fi%
  \def\@sizeName{b5r}
  \ifx\@paperSize\@sizeName\@paperWidth257mm\@paperHeight181mm\fi%
  \def\@sizeName{letterr}
  \ifx\@paperSize\@sizeName\@paperWidth11in\@paperHeight8.5in\fi%
  \def\@sizeName{envK2}		% 角2封筒
  \ifx\@paperSize\@sizeName\@paperWidth240mm\@paperHeight332mm\fi%
  \def\@sizeName{envT3}		% 長3封筒
  \ifx\@paperSize\@sizeName\@paperWidth120mm\@paperHeight235mm\fi%
  \def\@sizeName{envT3r}		% 長3封筒横
  \ifx\@paperSize\@sizeName\@paperWidth235mm\@paperHeight120mm\fi%
  \ifdim\@paperWidth=0pt\typein{unknown paper size : #1}\fi%
  #2
  % \topmargin = \topMargin - \headheight - \headsep
  \setlength{\topmargin}{\topMargin}%
  \addtolength{\topmargin}{-\headheight}%
  \addtolength{\topmargin}{-\headsep}%
  \addtolength{\topmargin}{-1in}%
  % \textheight = \@paperWidth - \topMargin - \bottomMargin
  \setlength{\textheight}{\@paperHeight}%
  \addtolength{\textheight}{-\topMargin}%
  \addtolength{\textheight}{-\bottomMargin}%
  % \oddsidemargin = \insideMargin
  \setlength{\oddsidemargin}{\insideMargin}%
  \addtolength{\oddsidemargin}{-1in}%
  % \evensidemargin = \outsideMargin
  \setlength{\evensidemargin}{\outsideMargin}%
  \addtolength{\evensidemargin}{-1in}%
  % \textwidth = \@paperWidth - \insideMargin - \outsideMargin
  \setlength{\textwidth}{\@paperWidth}%
  \addtolength{\textwidth}{-\insideMargin}
  \addtolength{\textwidth}{-\outsideMargin}
}

% マージン設定用の長さ変数
\newlength{\@paperWidth}
\newlength{\@paperHeight}
\newlength{\topMargin}
\newlength{\bottomMargin}
\newlength{\insideMargin}
\newlength{\outsideMargin}

% デフォルト用紙設定
\paperSize{a4}{%
  \topMargin 30mm
  \bottomMargin 30mm
  \insideMargin 25mm
  \outsideMargin 22mm
  \headheight 10mm
  \headsep 5mm
  \footskip 10mm
}

% ページスタイル
\pagestyle{plain}

% 行間調整
\renewcommand{\baselinestretch}{1.05}

\newif\ifmast

\def\bachelor{1}
\def\work{2}
\def\master{3}

\def\thesis#1{
  \ifnum#1=\bachelor
    \def\thesistitle{卒業研究報告書}
    \def\ethesistitle{Bachelor Thesis}
    \def\affiliation{大阪大学 工学部 電子情報工学科情報通信工学科目情報システム工学クラス}
    % \mastfalse の定義が必要
  \else
    \ifnum#1=\work
      \def\thesistitle{輪講レポート}
      \def\ethesistitle{report}
      \def\affiliation{大阪大学 工学部 電子情報工学科情報通信工学科目情報システム工学クラス}
      % \mastfalse の定義が必要
    \else
      \def\thesistitle{大学院博士前期課程修士学位論文}
      \def\ethesistitle{Master Thesis}
      \def\affiliation{大阪大学 大学院工学研究科 \\ 情報システム工学専攻}
      % \masttrue の定義が必要
    \fi
  \fi
}

\def\title#1{\gdef\@title{#1}}
\def\etitle#1{\gdef\@etitle{#1}}
\def\author#1{\gdef\@author{#1}}
\def\eauthor#1{\gdef\@eauthor{#1}}
\def\supervisor#1{\gdef\@supervisor{#1}}
\def\esupervisor#1{\gdef\@esupervisor{#1}}
\def\deadline#1{\gdef\@deadline{#1}}
\long\def\abstract#1{\long\gdef\@abstract{#1}}
\long\def\eabstract#1{\long\gdef\@eabstract{#1}}
\def\keyword#1{\gdef\@keyword{#1}}
\def\ekeyword#1{\gdef\@ekeyword{#1}}

\def\coverpage{%
\newpage\null
\thispagestyle{empty}
\setlength{\unitlength}{1mm}
\noindent\begin{picture}(163,232)(25,31)
 % 表紙は、電子情報工学科指定の様式。coverpage.eps を読込む
 \put(0,0){\includegraphics{coverpage.eps}}
 \put(60,189){\parbox[b]{92mm}{\centering\large\@title}}% 論文題目
 \put(105,150){\makebox(0,0)[b]{\large\@supervisor}}% 指導教員
 \put(105,106){\makebox(0,0)[b]{\large\@author}}% 著者
 \def\@extractdate 令和##1年##2月##3日{\def\@tempYear{##1}\def\@tempMonth{##2}\def\@tempDay{##3}}
 \expandafter\@extractdate\提出日
 \put(94.5,79.5){\makebox(0,0)[b]{\normalsize\@tempYear}}% 年
 \put(107,79.5){\makebox(0,0)[b]{\normalsize\@tempMonth}}% 月
 \put(120,79.5){\makebox(0,0)[b]{\normalsize\@tempDay}}% 日
\end{picture}%
\newpage
\if@twoside
\ \thispagestyle{empty}\newpage	% twosideの場合、1ページ空白ページ
\fi
\newpage
\pagenumbering{roman}
\begin{flushleft}\thesistitle\end{flushleft}
\vspace*{2mm}
\begin{flushleft}{\large \@title}\end{flushleft}
\begin{flushright}\@author\end{flushright}
\vspace*{2mm}
\begin{flushleft}{\gothic 内容梗概}\\ \end{flushleft}
\vspace*{2mm}

\@abstract
\vspace*{5mm}
\begin{flushleft}{\gothic キーワード}\\ \end{flushleft}
\@keyword
\ifmast
  \newpage
  \begin{flushleft}\ethesistitle\end{flushleft}
  \vspace*{2mm}
  \begin{flushleft}{\large \@etitle}\end{flushleft}
  \begin{flushright}\@eauthor\end{flushright}
  \vspace*{2mm}
  \begin{flushleft}{\bf Abstract}\\ \end{flushleft}
  \vspace*{2mm}
  \@eabstract
  \vspace*{5mm}
  \begin{flushleft}{\bf Keywords}\\ \end{flushleft}
  \@ekeyword
\fi
}

\newcommand{\body}{\newpage\pagenumbering{arabic}}
\newcommand{\acknowledgement}{
        \chapter*{謝辞\addcontentsline{toc}{chapter}{謝辞}}}
\def\thebibliography#1{\chapter*{参考文献\@mkboth
 {参 考 文 献}{参 考 文 献}\addcontentsline{toc}{chapter}{参考文献}}\list
 {[\arabic{enumi}]}{\settowidth\labelwidth{[#1]}\leftmargin\labelwidth
 \advance\leftmargin\labelsep
 \usecounter{enumi}}
 \def\newblock{\hskip .11em plus .33em minus .07em}
 \sloppy
 \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist
\def\appendix{\@ifnextchar[{\@appendix}{\@appendix[]}}
\def\@appendix[#1]{\chapter*{付録#1\addcontentsline{toc}{chapter}{付録#1}}}

\title{●}
\etitle{●}
\author{●}
\eauthor{●}
\supervisor{●}
\esupervisor{●}
\deadline{●年●月●日}
\abstract{●}
\eabstract{●}
\keyword{●}
\ekeyword{●}


