// Author(s): Rimco Boudewijns
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//

#include <QDateTime>
#include <QFileInfo>
#include <QDir>

#include "fileinformation.h"

FileInformation::FileInformation(QString filename, QWidget *parent) :
    QWidget(parent),
    m_filename(filename)
{
    m_ui.setupUi(this);

    QFileInfo info(filename);

    addRow("Name", info.fileName());
    addRow("Path", info.absoluteDir().absolutePath());
    addRow("Readable", (info.isReadable()?"Yes":"No"));
    addRow("Writable", (info.isExecutable()?"Yes":"No"));
    if (info.isFile())
    {
        addRow("Executable", (info.isExecutable()?"Yes":"No"));
        addRow("Date modified", info.lastModified().toString("yyyy-MM-dd hh:mm:ss"));
        addRow("Size", sizeString(info.size()));
    }
}

QString FileInformation::sizeString(qint64 size)
{
    double dsize = size;
    QStringList names;
    names << "bytes" << "kB" << "MB" << "GB" << "TB";
    int factor;
    for (factor = 0; factor < 4 && dsize >= 1024; factor++)
    {
      dsize /= 1024;
    }
    return QString("%1 %2").arg(dsize, 0, 'f', factor).arg(names.at(factor));
}

void FileInformation::addRow(QString name, QString value)
{
    QLabel *nameLabel = new QLabel(name.append(":"), this);
    QLabel *valueLabel = new QLabel(value, this);
    m_ui.formLayout->addRow(nameLabel, valueLabel);
}
