# -*- coding: utf-8 -*-
from odoo import models, fields, api, _
from odoo.exceptions import UserError
import base64
import csv
import io
from datetime import datetime


class HolidayImportWizard(models.TransientModel):
    _name = 'holiday.import.wizard'
    _description = 'Import Holiday Data'

    file = fields.Binary(string='CSV File', required=True)
    filename = fields.Char(string='Filename')
    delimiter = fields.Selection([
        (',', 'Comma (,)'),
        (';', 'Semicolon (;)'),
        ('\t', 'Tab')
    ], string='Delimiter', default=',', required=True)

    def action_import(self):
        """Import holiday data from CSV file"""
        if not self.file:
            raise UserError(_('Please select a file to import.'))

        # Decode the file content
        csv_data = base64.b64decode(self.file)
        csv_file = io.StringIO(csv_data.decode('utf-8'))

        # Create CSV reader
        reader = csv.DictReader(csv_file, delimiter=self.delimiter)

        # Check required fields
        required_fields = ['name', 'date']
        for field in required_fields:
            if field not in reader.fieldnames:
                raise UserError(_('The CSV file must contain the following fields: %s') % ', '.join(required_fields))

        # Process data
        holiday_data_obj = self.env['holiday.data']
        holidays_created = 0
        holidays_skipped = 0

        for row in reader:
            # Parse date
            try:
                date_str = row['date']
                # Try different date formats
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').date()
                except ValueError:
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                    except ValueError:
                        raise UserError(_('Invalid date format for %s: %s') % (row['name'], date_str))
            except Exception as e:
                raise UserError(_('Error parsing date: %s') % str(e))

            # Check if holiday already exists
            existing_holiday = holiday_data_obj.search([
                ('date', '=', date_obj),
                ('name', '=', row['name'])
            ])

            if existing_holiday:
                holidays_skipped += 1
                continue

            # Create holiday
            holiday_vals = {
                'name': row['name'],
                'date': date_obj,
                'description': row.get('description', ''),
                'type': row.get('type', ''),
                'country': row.get('country', ''),
                'year': int(row.get('year', date_obj.year))
            }

            holiday_data_obj.create(holiday_vals)
            holidays_created += 1

        # Show result message
        message = _('Import completed: %s holidays created, %s holidays skipped (already exist).') % (
        holidays_created, holidays_skipped)
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _('Holiday Import'),
                'message': message,
                'sticky': False,
                'type': 'success',
            }
        }