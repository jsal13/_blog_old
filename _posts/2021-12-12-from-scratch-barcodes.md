---
title:  "From Scratch: Barcodes"
date:   2021-12-12

description: Building up intuition for Barcodes.
categories: from_scratch

toc: true
excerpt: From Scratch is a series where we see how things work by slowly building up to them with simple, basic examples or toy models.  This time, Barcodes.
---

  *From Scratch* is a series where we see how things work by slowly building up to them with simple, basic examples or toy models.

---

## Introduction

Any motivation we come up with to invent our own "toy" version of barcodes will seem somewhat artificial since the original problem that lead to their invention — needing to quickly scan some small part of a package to get price and inventory information — had many potential solutions, and barcodes were, for whatever reason, the solution that stuck around for a while.  It's kind of a strange solution but it's one that worked and sometimes that's all that's needed.

*(An interesting note: it was originally intended for barcodes to be concentric circles instead of bars so they could be read in any direction by a scanner.)*

## Xs and Os: A Strange and Terrible Encoding

For the sake of simplicitly, let's say that we want to model a string of numbers that's made of 0s and 1s.  For example: 011101.  It would be easy to use binary here but let's make things a little weirder.  Let's make the following code with just Xs and Os.

    0 = oooo
    1 = ooox

We can now translate our original number into this strange X and O system.

    oooo ooox ooox ooox oooo oooo ooox
    0    1    1    1    0    0    1

Awesome.  We now have this verbose seemingly useless series of Xs and Os to represent the more simple 0 and 1.  What if we wanted to extend this a bit?  Let's try to write the numbers 0 to 9 in a similar, mostly arbitrary way:

    0 = oooo
    1 = ooox
    2 = ooxo
    3 = ooxx
    4 = oxoo
    5 = oxox
    6 = oxxo
    7 = oxxx
    8 = xooo
    9 = xoox

Once we have this, we can represent any sequence containing the numbers 0 to 9.  For example, 9414 becomes:

    xoox oxoo ooox oxoo
    9    4    1    4


## Ink & Warehouse Woes

Let's pretend our idea is so good that it gets adopted by a large warehouse our friend works at and they want to put our Xs and Os system on all of the crates depending on what they contain: the four-digit codes will be something like 1001 is for apples, 0431 is for phone cases, and so on.  But the guy who prints the labels — a surly gentleman named Pete — calls us up and tells us that printing all those Xs and Os is wasting a lot of ink.  We think for a moment and tell him:

"Okay, so long as we know each of the numbers will take up four spaces, and the spaces are the same size, what about we just put a vertical line for each X and leave the O space blank.  That should save on ink."

If we also remove the space between numbers that we've been putting above, our code for 9414 will now be:

    xooxoxoooooxoxoo
    |  | |     | |
    9   4   1   4

Wow, that will save a lot of ink!  Good for us.

Unfortunately, Pete calls back a week later and tells us they've been having a bit of trouble: sometimes the scanner isn't sure exactly where the code starts and ends, so sometimes people will scan 9414 but get 414 back.  That's a big problem!  What if we added a unique start code and a unique end code to tell the scanner where to start and end?

Sure, let's do that.

    xoxo   xxoo
    | |    ||
    start  end

The astute reader might, at this point, realize that certain combinations of letters and numbers may mimic how the start or end looks: this is true; for example, if we have something like 56 (oxox oxxo) then if the scanner skips the first 'o' it will see (xox oxxo) and if we don't have the space between the numbers this looks identical to the start code (xoxo).  Let's ignore this annoying point for now, but remember it for later.

Let's see what 9414 looks like.

    xoxoxooxoxoooooxoxooxxoo
    | | |  | |     | |  ||
    ST  9   4   1   4   END

## What about real barcodes?

This strange toy example is fairly close to the real thing.

![]({{ site.baseurl }}/images/barcode2.bmp "Sample Barcode")

Looking at this real barcode, you'll see some of the things we've done — sort of. These have the numbers they represent below them (not justified in the same way we had them, but they're there) and if you look at the left-hand side you'll see that each number (besides the 5 on the "outside") corresponds to a different set of four vertical lines and spaces.  You'll also notice that a number on the left won't have the same bars as a number on the right (for example, the 4 is different in this figure) — that is because *on many barcode types the left and right side have inverted bars-and-spaces*.  That is, if 4 on the left were something like xxox the 4 on the right would be ooxo.  Wild.

In addition, we see that there is a "starting" set of bars (the two bars after the first 5), a "middle" bar (after the 4, before the 1), as well as the "end" bar (after the 7).  This cuts the barcode into three main pieces: the number before the start bar, the numbers on the left side, and the numbers on the right side.  Each of these pieces will mean something different depending on the type of barcode it is.

Here's a few other caveats:

- There are many different types of barcodes which have many different encodings (i.e., the Xs and Os are in different places).
- The encodings often have certain restrictions: they're made so that you won't get a whole lot of blank space or a whole lot of thick vertical lines.  On a real product, you shouldn't see something which corresponds to a 4-thick vertical line.
- The numbers on the bottom correspond 1-to-1 with the bars on top, with the exception of the first number on the left, and the start-middle-end bars.
- Not all of the numbers are random: some may be checksums which check the validity of the barcode.

## That's all there is to know about Barcodes?

Not even close.  There's a ton of information out there on different barcodes, what those numbers mean, how long they should be, etc., and trying to do a high-level overview doesn't allow us to get into that kind of detail.

One interesting thing to think about is: how much information can we represent with barcodes?  Could we represent entire books?  Could we represent website URLs?  Could we represent music?  Video?

It's possible, but it's not going to be efficient if we stick to barcodes.  They're one-dimensional items that would need to be extremely long.  But what if we tried to extend this idea into two dimensions?  Hm.

![]({{ site.baseurl }}/images/qr.png "QR Code")
